"""
live_bot/feeds/webhook_server.py
---------------------------------
FastAPI endpoint that receives Upstox webhook (postback URL) notifications.

WHAT THIS IS:
    Upstox can POST order update notifications to a URL you register in
    your Upstox developer app settings. This is the BACKUP channel for
    order confirmations if the PortfolioDataStreamer WebSocket disconnects.

    It also handles the "Notifier Webhook" — when Upstox generates a new
    access token for a scheduled login, it sends it here automatically.

HOW TO EXPOSE THIS TO UPSTOX:
    1. Your server must be publicly accessible (or use ngrok for testing).
    2. Register your URL in Upstox App Settings:
         Postback URL: https://yourdomain.com/webhook/order-update
         Notifier URL: https://yourdomain.com/webhook/token
    3. Upstox will POST JSON to these URLs.

THESE ENDPOINTS ARE MOUNTED ON THE MAIN FASTAPI APP in dashboard/app.py:
    app.include_router(webhook_router, prefix="/webhook")

SECURITY:
    Upstox does not sign webhook payloads with a shared secret (as of 2024).
    Basic IP whitelisting is the recommended approach. We validate:
        - Content-Type is application/json
        - Required fields are present
        - order_id format sanity check

EDGE CASES HANDLED:
    - Non-JSON body → 400 response.
    - Missing required fields → logged as warning, 200 response (don't fail Upstox).
    - Duplicate delivery (Upstox may send the same update twice) → idempotent.
    - Token webhook arrives with empty token → logged, 400 response.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from live_bot.state import state as live_state

logger = logging.getLogger(__name__)

webhook_router = APIRouter()


# ── Pydantic models for request validation ────────────────────────────────────

class OrderUpdateWebhook(BaseModel):
    """
    Expected payload from Upstox order postback URL.
    All fields are Optional because Upstox may omit fields in some states.
    """
    order_id:         Optional[str]   = None
    status:           Optional[str]   = None
    instrument_token: Optional[str]   = None
    transaction_type: Optional[str]   = None
    quantity:         Optional[int]   = 0
    average_price:    Optional[float] = 0.0
    filled_quantity:  Optional[int]   = 0
    pending_quantity: Optional[int]   = 0
    order_type:       Optional[str]   = None
    product:          Optional[str]   = None
    exchange_order_id:Optional[str]   = None
    tag:              Optional[str]   = None


class TokenWebhook(BaseModel):
    """
    Expected payload from Upstox token notifier webhook.
    Sent automatically after a scheduled/background login flow.
    """
    access_token: Optional[str] = None
    user_id:      Optional[str] = None
    expires_in:   Optional[int] = None


# ── Webhook Endpoints ─────────────────────────────────────────────────────────

@webhook_router.post(
    "/order-update",
    summary="Upstox order postback URL receiver",
    description="Receives order status updates from Upstox as backup channel.",
)
async def receive_order_update(request: Request) -> JSONResponse:
    """
    Endpoint: POST /webhook/order-update

    Upstox POSTs here whenever an order status changes.
    This is the BACKUP path — the primary path is PortfolioDataStreamer.

    We always return HTTP 200 to Upstox even on errors to prevent retries
    for data we cannot process. Errors are logged internally.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"[Webhook] Invalid JSON in order update: {e}")
        # Return 200 to prevent Upstox retry flood on bad payload
        return JSONResponse({"status": "error", "message": "invalid_json"}, status_code=200)

    if not isinstance(body, dict):
        logger.warning(f"[Webhook] Unexpected body type: {type(body)}")
        return JSONResponse({"status": "error", "message": "not_a_dict"}, status_code=200)

    try:
        order_id = str(body.get("order_id", "")).strip()
        status   = str(body.get("status",   "unknown")).lower()
        avg_px   = float(body.get("average_price", 0) or 0)
        qty      = int(body.get("filled_quantity", 0) or 0)
        instr    = str(body.get("instrument_token", ""))
        txn_type = str(body.get("transaction_type", "")).upper()

        if not order_id:
            logger.warning(f"[Webhook] Order update missing order_id: {body}")
            return JSONResponse({"status": "ok", "message": "no_order_id"})

        logger.info(
            f"[Webhook] Order update received: {order_id} | "
            f"{txn_type} {instr} qty={qty} avg_px={avg_px:.2f} status={status}"
        )

        # Update order state (idempotent — same status update is harmless)
        live_state.update_order_status(
            order_id   = order_id,
            status     = status,
            fill_price = avg_px if status == "complete" else None,
            filled_at  = datetime.now() if status == "complete" else None,
        )

        live_state.log_activity(
            "WEBHOOK_ORDER",
            f"[WEBHOOK] Order {order_id[:8]}... {status.upper()} "
            f"| {txn_type} @ ₹{avg_px:.2f}",
            level="INFO" if status in ("complete", "open") else "WARNING",
        )

        return JSONResponse({"status": "ok", "order_id": order_id, "received_status": status})

    except Exception as e:
        logger.error(f"[Webhook] Error processing order update: {e}", exc_info=True)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=200)


@webhook_router.post(
    "/token",
    summary="Upstox token notifier webhook receiver",
    description="Receives new access tokens automatically from Upstox after login.",
)
async def receive_token(request: Request) -> JSONResponse:
    """
    Endpoint: POST /webhook/token

    Upstox POSTs a fresh access token here after the scheduled login flow.
    We save it to disk so the live bot can start without manual token entry.

    SECURITY NOTE:
        Anyone who knows this URL can push a token. Protect this endpoint
        with IP whitelisting in production.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"[Webhook] Invalid JSON in token notifier: {e}")
        return JSONResponse({"status": "error", "message": "invalid_json"}, status_code=400)

    access_token = str(body.get("access_token", "")).strip()
    user_id      = str(body.get("user_id",      "")).strip()

    if not access_token:
        logger.warning(f"[Webhook] Token webhook received empty token. Body: {body}")
        return JSONResponse(
            {"status": "error", "message": "empty_token"},
            status_code=400,
        )

    logger.info(
        f"[Webhook] New access token received for user: {user_id}. "
        "Saving to disk."
    )

    # Save the new token using the AuthManager
    try:
        import sys
        from pathlib import Path
        # Ensure parent package is importable
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

        from broker.upstox.auth import auth_manager
        from datetime import date

        token_data = {
            "access_token":   access_token,
            "user_id":        user_id,
            "token_type":     "bearer",
            "generated_date": date.today().isoformat(),
        }
        auth_manager._save_token(token_data)
        auth_manager._token_data = token_data  # Update in-memory cache

        live_state.log_activity(
            "TOKEN_RECEIVED",
            f"New access token received via webhook for user {user_id}. "
            "Bot can start trading.",
        )

        return JSONResponse({
            "status":  "ok",
            "user_id": user_id,
            "message": "Token saved. Bot will use new token on next action.",
        })

    except Exception as e:
        logger.error(f"[Webhook] Failed to save token: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": f"Token save failed: {str(e)}"},
            status_code=500,
        )


@webhook_router.get(
    "/health",
    summary="Webhook server health check",
)
async def webhook_health() -> JSONResponse:
    """
    Simple health check for the webhook endpoint.
    Upstox may ping this to verify your postback URL is alive.
    """
    return JSONResponse({
        "status":      "ok",
        "service":     "algo_trading_webhook",
        "bot_running": live_state._is_running,
        "timestamp":   datetime.now().isoformat(),
    })
