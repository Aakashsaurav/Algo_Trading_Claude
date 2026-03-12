from live_bot.engine import LiveBotEngine, LiveBotConfig
from strategies.momentum.ema_crossover import EMACrossoverStrategy
from broker.upstox.auth import auth_manager

token = auth_manager.get_valid_token()

config = LiveBotConfig(
    strategy_class   = EMACrossoverStrategy,
    strategy_params  = {"fast_period": 9, "slow_period": 21},
    instrument_map   = {
        "NSE_EQ|INE020B01018": "RELIANCE",
        "NSE_EQ|INE040A01034": "HDFCBANK",
        "NSE_EQ|INE467B01029": "TCS",
    },
    initial_capital  = 500_000,
    product          = "I",           # MIS intraday
    daily_loss_limit_pct = 2.0,
)

bot = LiveBotEngine(config, token)
bot.start()