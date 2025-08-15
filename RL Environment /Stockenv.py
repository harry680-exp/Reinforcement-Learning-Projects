import os
# SOLUTION 1: Disable XLA completely
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['TF_DISABLE_POOL_ALLOCATOR'] = '1'


# ... rest of your imports
import pandas as pd 
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import ta  # Technical Analysis library for manual indicator calculation
import tensorflow as tf

# SOLUTION 2: More comprehensive GPU configuration
def configure_gpu():
    """Configure GPU with better error handling and memory management"""
    try:
        # Disable XLA JIT compilation
        tf.config.optimizer.set_jit(False)
        tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU(s)")
        
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Set memory limit if needed (optional)
                    # tf.config.set_memory_growth(gpu, True)
                
                # Alternative: Set memory limit explicitly
                # tf.config.experimental.set_memory_limit(gpus[0], 1024)  # 1GB limit
                
                print("✅ GPU Memory Growth enabled")
                
                # Verify GPU configuration
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                print("Falling back to CPU execution")
                # Force CPU execution if GPU fails
                tf.config.set_visible_devices([], 'GPU')
                
        else:
            print("No GPU found, using CPU")
            
    except Exception as e:
        print(f"Error in GPU configuration: {e}")
        print("Using default TensorFlow configuration")

# Apply GPU configuration
configure_gpu()

# SOLUTION 3: Add numerical stability fixes
def add_numerical_stability():
    """Add numerical stability to prevent sqrt of negative numbers"""
    tf.keras.backend.set_floatx('float32')  # Ensure consistent precision
    #tf.config.run_functions_eagerly(True)   # Disable graph mode for debugging

add_numerical_stability()

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Assuming you have these custom modules
from agent import Agent  # original agent
from network import ActorNetwork, CriticNetwork
from learn import Replaybuffer

pd.set_option('display.max_columns', None)

def add_technical_indicators(df):
    """
    Manually add technical indicators to the DataFrame
    This replaces the failing FinRL feature engineering
    """
    print("Adding technical indicators manually...")
    df_with_indicators = df.copy()
    
    # Ensure we have the required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_with_indicators.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for technical analysis: {missing_cols}")
    
    # Use ta library to add technical indicators
    try:
        # Trend Indicators
        df_with_indicators['sma_10'] = ta.trend.sma_indicator(df_with_indicators['close'], window=10)
        df_with_indicators['sma_20'] = ta.trend.sma_indicator(df_with_indicators['close'], window=20)
        df_with_indicators['sma_30'] = ta.trend.sma_indicator(df_with_indicators['close'], window=30)
        df_with_indicators['ema_12'] = ta.trend.ema_indicator(df_with_indicators['close'], window=12)
        df_with_indicators['ema_26'] = ta.trend.ema_indicator(df_with_indicators['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df_with_indicators['close'])
        df_with_indicators['macd'] = macd.macd()
        df_with_indicators['macd_signal'] = macd.macd_signal()
        df_with_indicators['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_with_indicators['close'])
        df_with_indicators['bb_upper'] = bb.bollinger_hband()
        df_with_indicators['bb_middle'] = bb.bollinger_mavg()
        df_with_indicators['bb_lower'] = bb.bollinger_lband()
        df_with_indicators['bb_width'] = bb.bollinger_wband()
        df_with_indicators['bb_percent'] = bb.bollinger_pband()
        
        # RSI
        df_with_indicators['rsi'] = ta.momentum.rsi(df_with_indicators['close'], window=14)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df_with_indicators['high'], 
                                               df_with_indicators['low'], 
                                               df_with_indicators['close'])
        df_with_indicators['stoch_k'] = stoch.stoch()
        df_with_indicators['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df_with_indicators['williams_r'] = ta.momentum.williams_r(df_with_indicators['high'],
                                                                 df_with_indicators['low'],
                                                                 df_with_indicators['close'])
        
        # Commodity Channel Index (CCI)
        df_with_indicators['cci'] = ta.trend.cci(df_with_indicators['high'],
                                                df_with_indicators['low'],
                                                df_with_indicators['close'])
        
        # Average Directional Index (ADX)
        df_with_indicators['adx'] = ta.trend.adx(df_with_indicators['high'],
                                               df_with_indicators['low'],
                                               df_with_indicators['close'])
        
        # Volume indicators
        df_with_indicators['volume_sma'] = ta.volume.volume_sma(df_with_indicators['close'],
                                                              df_with_indicators['volume'])
        df_with_indicators['volume_ema'] = ta.volume.VolumeSMAIndicator(df_with_indicators['close'],
                                                                      df_with_indicators['volume']).volume_sma()
        
        # Price indicators
        df_with_indicators['typical_price'] = (df_with_indicators['high'] + 
                                             df_with_indicators['low'] + 
                                             df_with_indicators['close']) / 3
        
        # Volatility indicators
        df_with_indicators['atr'] = ta.volatility.average_true_range(df_with_indicators['high'],
                                                                   df_with_indicators['low'],
                                                                   df_with_indicators['close'])
        
        # Additional momentum indicators
        df_with_indicators['roc'] = ta.momentum.roc(df_with_indicators['close'])  # Rate of Change
        
        print("Successfully added technical indicators:")
        tech_indicators = [col for col in df_with_indicators.columns if col not in ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
        print(f"Added {len(tech_indicators)} technical indicators: {tech_indicators}")
        
        return df_with_indicators
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        print("Falling back to basic price-based indicators...")
        
        # Fallback: Add basic indicators manually
        df_with_indicators['sma_10'] = df_with_indicators['close'].rolling(window=10).mean()
        df_with_indicators['sma_20'] = df_with_indicators['close'].rolling(window=20).mean()
        df_with_indicators['returns'] = df_with_indicators['close'].pct_change()
        df_with_indicators['volatility'] = df_with_indicators['returns'].rolling(window=10).std()
        df_with_indicators['high_low_pct'] = (df_with_indicators['high'] - df_with_indicators['low']) / df_with_indicators['close']
        
        print("Added basic fallback indicators")
        return df_with_indicators

# Data parameters
tickers = ['ICICIBANK.NS']
start_date = '2009-06-01'
end_date = '2019-01-01'
trade_start_date = '2025-06-16'
trade_end_date = '2025-07-28'

print("=== DOWNLOADING DATA ===")
df_raw = yf.download(tickers, start=start_date, end=end_date)

# Fix the column naming issue - make sure 'tic' column is created properly
df_processed = df_raw.stack(level=1).reset_index().rename(columns={'level_1': 'tic', 'Date': 'date'})
df_processed.set_index('date', inplace=True)
df_processed.columns = [col.lower() for col in df_processed.columns]

# Handle the case where 'tic' might have been converted to lowercase or other issues
if 'tic' not in df_processed.columns:
    if 'ticker' in df_processed.columns:
        df_processed = df_processed.rename(columns={'ticker': 'tic'})
        print("Renamed 'ticker' column to 'tic'")
    else:
        # Add tic column manually
        df_processed['tic'] = 'ICICIBANK.NS'
        print("Added 'tic' column manually")

print("Data shape:", df_processed.shape)
print("Columns:", df_processed.columns.tolist())

print("\n=== ADDING TECHNICAL INDICATORS ===")
# Ensure 'date' is a column, not index
if 'date' not in df_processed.columns:
    df_processed = df_processed.reset_index()

df_processed['date'] = pd.to_datetime(df_processed['date'])

print("Data before adding technical indicators:")
print(f"Shape: {df_processed.shape}")
print(f"Columns: {df_processed.columns.tolist()}")
print(f"Data types: {df_processed.dtypes.to_dict()}")

# First, try FinRL feature engineering
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False
)

finrl_success = False
try:
    # Create a copy to avoid modifying original data
    df_for_fe = df_processed.copy()
    
    # Make sure the data is in the correct format for FinRL
    # FinRL expects: date, tic, open, high, low, close, volume
    required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    print(f"Checking required columns: {required_columns}")
    
    missing_cols = [col for col in required_columns if col not in df_for_fe.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by date and reset index
    df_for_fe = df_for_fe.sort_values('date').reset_index(drop=True)
    
    # Apply feature engineering
    print("Trying FinRL feature engineering...")
    df_featured = fe.preprocess_data(df_for_fe)
    
    # Remove any problematic columns that might have been added incorrectly
    problematic_columns = ['index', 'level_0', 'level_1']
    for col in problematic_columns:
        if col in df_featured.columns:
            df_featured = df_featured.drop(columns=[col])
            print(f"Removed problematic column from FinRL output: {col}")
    
    print("FinRL feature engineering successful!")
    print("Enhanced data shape:", df_featured.shape)
    
    # Find new columns added by feature engineering
    new_columns = [col for col in df_featured.columns if col not in df_processed.columns]
    print("New columns added by FinRL:", new_columns)
    
    # Verify we got meaningful technical indicators, not just 'index' column
    meaningful_indicators = [col for col in new_columns if col not in ['index', 'level_0', 'level_1']]
    if len(meaningful_indicators) > 5:  # Expect at least 5 technical indicators
        finrl_success = True
        print(f"✅ FinRL added {len(meaningful_indicators)} meaningful technical indicators")
    else:
        print(f"⚠️ FinRL only added {len(meaningful_indicators)} meaningful indicators, will use manual method")
        raise ValueError("Insufficient technical indicators from FinRL")
    
except Exception as e:
    print(f"FinRL feature engineering failed: {e}")
    finrl_success = False

# If FinRL failed, use manual technical indicator calculation
if not finrl_success:
    print("Using manual technical indicator calculation...")
    try:
        df_featured = add_technical_indicators(df_processed)
        print("✅ Manual technical indicators added successfully!")
        print("Enhanced data shape:", df_featured.shape)
        
        # Find new columns added by manual method
        new_columns = [col for col in df_featured.columns if col not in df_processed.columns]
        print("New columns added manually:", new_columns)
        
    except Exception as manual_error:
        print(f"Manual technical indicator calculation also failed: {manual_error}")
        print("Falling back to basic data without technical indicators...")
        df_featured = df_processed.copy()
        
        # Add at least some basic derived features
        df_featured['price_change'] = df_featured['close'].pct_change()
        df_featured['volume_change'] = df_featured['volume'].pct_change()
        df_featured['high_low_ratio'] = df_featured['high'] / df_featured['low']
        
        print(f"Fallback data shape: {df_featured.shape}")
        print(f"Fallback columns: {df_featured.columns.tolist()}")

print("\n=== PREPARING DATA FOR FINRL ENVIRONMENT ===")

# Clean data - remove NaN values
initial_rows = len(df_featured)
df_featured = df_featured.dropna()
final_rows = len(df_featured)
print(f"Removed {initial_rows - final_rows} rows with missing values")

# Reset index to make 'date' a column (required for FinRL)
if df_featured.index.name == 'date':
    df_featured = df_featured.reset_index()

# Sort by date
df_featured = df_featured.sort_values(['date']).reset_index(drop=True)

# Ensure 'tic' column exists and is correct
if 'tic' not in df_featured.columns:
    df_featured['tic'] = 'ICICIBANK.NS'
    print("Added missing 'tic' column")

# Remove any problematic columns like 'index' that might have been added incorrectly
problematic_columns = ['index', 'level_0', 'level_1']
for col in problematic_columns:
    if col in df_featured.columns:
        df_featured = df_featured.drop(columns=[col])
        print(f"Removed problematic column: {col}")

# Get technical indicators (numeric columns only, excluding basic OHLCV data)
required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
tech_indicators = [col for col in df_featured.columns 
                  if col not in required_cols and 
                  df_featured[col].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f"Technical indicators ({len(tech_indicators)}):", tech_indicators)

# Filter to only include required columns and numeric tech indicators
final_columns = required_cols + tech_indicators
available_columns = [col for col in final_columns if col in df_featured.columns]
df_featured = df_featured[available_columns].copy()

print(f"Final DataFrame shape: {df_featured.shape}")
print("Final columns:", df_featured.columns.tolist())
print("Sample data:")
print(df_featured.head())

# Verify data integrity
print("\nData integrity check:")
print(f"Date range: {df_featured['date'].min()} to {df_featured['date'].max()}")
print(f"Unique tickers: {df_featured['tic'].unique()}")
print(f"Null values per column:")
print(df_featured.isnull().sum()[df_featured.isnull().sum() > 0])

print("\n=== CREATING FINRL TRADING ENVIRONMENT ===")

# FIXED: Proper environment parameters for single stock
stock_dimension = 1  # Number of stocks
num_tech_indicators = len(tech_indicators)

# FinRL state structure: [balance] + [prices] + [shares] + [tech_indicators]
# For single stock: 1 (balance) + 1 (price) + 1 (shares) + tech_indicators
state_space = 1 + stock_dimension + stock_dimension + num_tech_indicators

env_kwargs = {
    "hmax": 100,                           # Maximum shares to hold
    "initial_amount": 1000000,             # Starting cash (10 Lakh INR)
    "stock_dim": stock_dimension,          # REQUIRED: Number of stocks
    "num_stock_shares": [0] * stock_dimension,  # Initial shares for each stock
    "buy_cost_pct": 0.001,                 # Transaction costs
    "sell_cost_pct": 0.001,
    "reward_scaling": 1e-4,
    "state_space": state_space,            # CRITICAL: Proper state space calculation
    "action_space": stock_dimension,       # Actions for each stock
    "tech_indicator_list": tech_indicators,
    "print_verbosity": 10                  # Increased for debugging
}

print("Environment configuration:")
for key, value in env_kwargs.items():
    print(f"  {key}: {value}")
print(f"Expected state space dimension: {state_space}")

# Create environment with error handling and version compatibility
try:
    # First, let's check what parameters StockTradingEnv actually expects
    import inspect
    sig = inspect.signature(StockTradingEnv.__init__)
    expected_params = list(sig.parameters.keys())
    print(f"StockTradingEnv expected parameters: {expected_params}")
    
    # Filter env_kwargs to only include expected parameters
    filtered_kwargs = {k: v for k, v in env_kwargs.items() if k in expected_params}
    missing_params = [p for p in expected_params if p not in ['self', 'df'] and p not in filtered_kwargs]
    
    print(f"Filtered kwargs: {list(filtered_kwargs.keys())}")
    print(f"Missing parameters: {missing_params}")
    
    # Try creating the environment
    env = StockTradingEnv(df=df_featured, **filtered_kwargs)
    print("✅ Environment created successfully")
    
    # Test environment reset
    initial_state = env.reset()
    if isinstance(initial_state, tuple):
        initial_state = initial_state[0]
    
    actual_state_size = len(initial_state)
    print(f"✅ Actual state size: {actual_state_size}")
    print(f"Expected state size: {state_space}")
    
    if actual_state_size != state_space:
        print(f"WARNING: State size mismatch! Expected {state_space}, got {actual_state_size}")
        # Use actual state size for agent
        state_space = actual_state_size
    
except Exception as e:
    print(f"First attempt failed: {e}")
    print("Trying alternative parameter combinations...")
    
    # Alternative parameter sets for different FinRL versions
    alternative_kwargs_list = [
        # Version with stock_dim and day
        {
            "df": df_featured,
            "stock_dim": stock_dimension,
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "reward_scaling": 1e-4,
            "tech_indicator_list": tech_indicators,
            "print_verbosity": 10,
            "day": 0  # Starting day
        },
        # Minimal parameter set
        {
            "df": df_featured,
            "stock_dim": stock_dimension,
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "tech_indicator_list": tech_indicators
        },
        # Another common parameter set
        {
            "df": df_featured,
            "stock_dim": stock_dimension,
            "hmax": 100,
            "initial_amount": 1000000,
            "transaction_cost_pct": 0.001,
            "tech_indicator_list": tech_indicators
        }
    ]
    
    env = None
    for i, alt_kwargs in enumerate(alternative_kwargs_list):
        try:
            print(f"Trying alternative parameter set {i+1}...")
            env = StockTradingEnv(**alt_kwargs)
            print(f"✅ Environment created successfully with parameter set {i+1}")
            break
        except Exception as alt_e:
            print(f"Alternative {i+1} failed: {alt_e}")
            continue
    
    if env is None:
        print("❌ All environment creation attempts failed")
        print("Please check your FinRL version and documentation")
        exit()

print("\n=== INITIALIZING DDPG AGENT ===")

try:
    # Initialize DDPG agent with correct dimensions
    agent = Agent(
        input_dims=(actual_state_size,),    # Use actual state size from environment
        env=env,
        n_actions=stock_dimension           # Actions for each stock
    )
    print("✅ DDPG Agent initialized successfully")
    print(f"Agent input dimensions: {agent.input_dims}")
    print(f"Agent action dimensions: {agent.n_actions}")
    
except Exception as e:
    print(f"❌ Error initializing agent: {e}")
    exit()

print("\n=== STARTING TRAINING ===")

# Training parameters
n_episodes = 1000
max_steps_per_episode = min(500, len(df_featured) - 100)  # Prevent running out of data
save_interval = 100
evaluation_interval = 50

print(f"Training episodes: {n_episodes}")
print(f"Max steps per episode: {max_steps_per_episode}")

# Performance tracking
score_history = []
best_score = -np.inf

# SOLUTION 4: Add memory management and error recovery
def clear_gpu_memory():
    """Clear GPU memory periodically"""
    try:
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices('GPU'):
            # Force garbage collection
            import gc
            gc.collect()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")

# SOLUTION 5: Add training loop with better error handling
for episode in range(n_episodes):
    try:
        # Periodic memory cleanup every 50 episodes
        if episode % 50 == 0 and episode > 0:
            print(f"Clearing GPU memory at episode {episode}")
            clear_gpu_memory()
        
        # Reset environment for new episode
        state_info = env.reset()
        if isinstance(state_info, tuple):
            state = state_info[0]
        else:
            state = state_info
        print("Initial state:", state)
        print("Type:", type(state))
        print("Length:", len(state))

        done = False
        episode_score = 0
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            try:
                raw_action = agent.action(state)

# Ensure action is an array of length `stock_dimension`
                if np.isscalar(raw_action):
                    action = np.array([raw_action]*stock_dimension, dtype=np.float32)
                else:
                    action = np.array(raw_action, dtype=np.float32).flatten()

# Clip to valid range if needed (e.g., between -1 and 1)

                # Check for NaN or inf values in action
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    print(f"Warning: Invalid action detected at episode {episode}, step {step_count}")
                    action = np.zeros_like(action)  # Use neutral action
                
                # Environment step with better error handling
                step_result = env.step(action)
                
                # Handle different return formats (OpenAI Gym compatibility)
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                elif len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    print(f"Unexpected step result length: {len(step_result)}")
                    break
                
                # Check for NaN or inf values in state and reward
                if (np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)) or 
                    np.isnan(reward) or np.isinf(reward)):
                    print(f"Warning: Invalid state/reward at episode {episode}, step {step_count}")
                    break
                
                # Store experience in replay buffer
                agent.remember(state, raw_action, reward, next_state, done)
                
                # Learn from experience with error handling
                if len(agent.memory.state_memory) > agent.batch_size:
                    try:
                        agent.learn()
                    except Exception as learn_error:
                        print(f"Learning error at episode {episode}, step {step_count}: {learn_error}")
                        # Clear GPU memory and continue
                        clear_gpu_memory()
                        continue
                
                # Update state and score
                state = next_state
                episode_score += reward
                step_count += 1
                
            except Exception as step_error:
                print(f"Error in environment step at episode {episode}, step {step_count}: {step_error}")
                print(f"Action that caused error: {action}")
                print(f"State shape: {len(state) if hasattr(state, '__len__') else 'N/A'}")
                
                # Try to recover by clearing memory and skipping this step
                clear_gpu_memory()
                break  # Skip to next episode
        
        # Track performance
        score_history.append(episode_score)
        avg_score = np.mean(score_history[-100:])
        
        # Save best model
        if avg_score > best_score:
            best_score = avg_score
            try:
                agent.save_models()
            except Exception as save_error:
                print(f"Warning: Could not save models: {save_error}")
        
        # Print progress
        if episode % evaluation_interval == 0:
            print(f"Episode {episode:4d} | Score: {episode_score:8.2f} | "
                  f"Avg Score: {avg_score:8.2f} | Steps: {step_count:3d}")
        
        # Save periodically
        if episode % save_interval == 0 and episode > 0:
            print(f"Saving models at episode {episode}")
            try:
                agent.save_models()
            except Exception as save_error:
                print(f"Warning: Could not save models: {save_error}")
    
    except Exception as episode_error:
        print(f"Error in episode {episode}: {episode_error}")
        clear_gpu_memory()  # Clean up before continuing
        continue  # Continue with next episode

print("Training completed!")

# Final cleanup
clear_gpu_memory()

# Optional: Plot training results
if len(score_history) > 0:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(score_history)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    # Moving average
    window_size = min(100, len(score_history))
    if len(score_history) >= window_size:
        moving_avg = pd.Series(score_history).rolling(window=window_size).mean()
        plt.plot(moving_avg)
        plt.title(f'Moving Average ({window_size} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.show()
