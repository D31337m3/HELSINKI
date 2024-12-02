#!/usr/bin/env python3
"""
Helsinki - Crypto Lottery Seed Generator
Version 1.0.1

Core Features:
- Multi-chain wallet generation using cryptographic primitives
- Real-time balance checking across multiple networks
- Secure API key management with validation
- Performance monitoring and metrics collection
- Comprehensive logging with detailed statistics

Technical Implementation:
- Uses BIP39 for mnemonic generation
- Implements token bucket algorithm for rate limiting
- Exponential backoff for API call resilience
- Memory management for long-running sessions
- Thread-safe operations for concurrent processing
"""

import binascii
import json
import os
import gc
import time
import functools
from pathlib import Path
from typing import Dict, Any, Callable
from dataclasses import dataclass
import argparse
import time
import secrets
import hashlib
import binascii
import json
import os
import urllib.request
import urllib.error
import logging
from datetime import datetime
from typing import Tuple, List, Dict
from pathlib import Path

# Core system constants with type hints for better code clarity
WORD_LIST_PATH: str = "wordlist.txt"  # BIP39 wordlist location
LOGS_DIR: Path = Path("logs")         # Directory for storing operation logs
CONFIG_FILE: Path = Path("config.txt") # User configuration storage
API_DELAY: float = 1.0                # Default delay between API calls
ATTEMPT_COUNTER: int = 0              # Tracks total generation attempts
VERSION: str = "1.0.1"                # Current software version
MAX_RETRIES: int = 3                  # Maximum API retry attempts
REQUEST_TIMEOUT: int = 10             # API request timeout in seconds
DEFAULT_MAX_ATTEMPTS: int = 50        # Default maximum attempts for balance checking

def initialize_environment():
    """Initialize required files and folders for Helsinki"""
    # Ensure logs directory exists, if not, create it
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(parents=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Ensure config file exists, if not, create an empty one
    if not CONFIG_FILE.exists():
        ConfigManager.save({})
        
    # Download wordlist if not present
    if not Path(WORD_LIST_PATH).exists():
        print("Downloading BIP39 wordlist...")
        wordlist_url = "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt"
        urllib.request.urlretrieve(wordlist_url, WORD_LIST_PATH)
        print("Wordlist downloaded successfully")

class Network:
    """
    Network configuration handler
    
    Attributes:
        name: Network identifier (e.g., 'ETH', 'BTC')
        api_url: Base URL for API endpoints
        api_key: Authentication key for API access
        decimals: Token decimal places for balance calculation
        chain_id: Network chain identifier
    """
    def __init__(self, name: str, api_url: str, api_key: str = None, decimals: int = 18, chain_id: int = None):
        self.name = name
        self.api_url = api_url.rstrip('/')  # Ensure consistent URL format
        self.api_key = api_key
        self.decimals = decimals
        self.chain_id = chain_id

@dataclass
class NetworkConfig:
    name: str
    api_url: str
    chain_id: int
    decimals: int
class NetworkFactory:
          @staticmethod
          def create_networks(config_path: Path) -> Dict[str, Network]:
              """
              Creates network configurations from JSON instead of YAML
              """
              with open(config_path, 'r') as f:
                  network_configs = json.load(f)
            
              networks = {}
              for name, config in network_configs.items():
                  networks[name] = Network(
                      name=config['name'],
                      api_url=config['api_url'],
                      chain_id=config['chain_id'],
                      decimals=config.get('decimals', 18)
                  )
              return networks

class RateLimiter:
    def __init__(self, calls_per_second: int):
        self.rate = calls_per_second
        self.tokens = self.rate
        self.last_update = time.time()
        self.token_bucket_size = calls_per_second * 2

    def acquire(self) -> None:
        current = time.time()
        time_passed = current - self.last_update
        self.tokens = min(
            self.token_bucket_size,
            self.tokens + time_passed * self.rate
        )
        
        if self.tokens < 1:
            sleep_time = (1 - self.tokens) / self.rate
            time.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1
        
        self.last_update = current

class MemoryManager:
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.initial_usage = self._get_memory_usage()
  
    def _get_memory_usage(self) -> int:
        """
        Get memory usage using native Python memory tracking
        """
        import sys
        return sys.getsizeof(globals()) + sys.getsizeof(locals())
      
    def memory_managed_check(self) -> None:
        if self._get_memory_usage() > self.threshold_bytes:
            gc.collect()
      
    def get_memory_usage(self) -> Dict[str, float]:
        current_usage = self._get_memory_usage()
        return {
            'current_mb': current_usage / 1024 / 1024,
            'delta_mb': (current_usage - self.initial_usage) / 1024 / 1024
        }
    
    class RetryStrategy:
        
         @staticmethod
         def with_exponential_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                return None
            return wrapper
    def with_exponential_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                return None
            return wrapper
class NetworkFactory:
      @staticmethod
      def create_networks(config_path: Path) -> Dict[str, Network]:
          """
          Creates network configurations from JSON instead of YAML
          """
          with open(config_path, 'r') as f:
              network_configs = json.load(f)
            
          networks = {}
          for name, config in network_configs.items():
              networks[name] = Network(
                  name=config['name'],
                  api_url=config['api_url'],
                  chain_id=config['chain_id'],
                  decimals=config.get('decimals', 18)
              )
          return networks

class MemoryManager:
      def __init__(self, threshold_mb: int = 1000):
          self.threshold_bytes = threshold_mb * 1024 * 1024
          self.initial_usage = self._get_memory_usage()
        
      def _get_memory_usage(self) -> int:
          """
          Get memory usage using native Python memory tracking
          """
          import sys
          return sys.getsizeof(globals()) + sys.getsizeof(locals())
            
      def memory_managed_check(self) -> None:
          if self._get_memory_usage() > self.threshold_bytes:
              gc.collect()
            
      def get_memory_usage(self) -> Dict[str, float]:
          current_usage = self._get_memory_usage()
          return {
              'current_mb': current_usage / 1024 / 1024,
              'delta_mb': (current_usage - self.initial_usage) / 1024 / 1024
          }

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'attempts': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'api_latencies': [],
            'start_time': time.time()
        }

    def track_performance(self, attempts: int, duration: float) -> None:
        self.metrics['attempts'] = attempts
        self.metrics['api_latencies'].append(duration)
        
    def track_api_call(self, success: bool, latency: float) -> None:
        if success:
            self.metrics['successful_calls'] += 1
        else:
            self.metrics['failed_calls'] += 1
        self.metrics['api_latencies'].append(latency)
        
    def get_statistics(self) -> Dict[str, Any]:
        total_time = time.time() - self.metrics['start_time']
        avg_latency = (
            sum(self.metrics['api_latencies']) / 
            len(self.metrics['api_latencies'])
            if self.metrics['api_latencies'] else 0
        )
        
        return {
            'total_attempts': self.metrics['attempts'],
            'success_rate': (
                self.metrics['successful_calls'] /
                (self.metrics['successful_calls'] + self.metrics['failed_calls'])
                if (self.metrics['successful_calls'] + self.metrics['failed_calls']) > 0
                else 0
            ),
            'average_latency': avg_latency,
            'attempts_per_second': self.metrics['attempts'] / total_time if total_time > 0 else 0
        }

# First, create the logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Then configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'helsinki.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
class Network:
    """Enhanced network configuration with validation"""
    def __init__(self, name: str, api_url: str, api_key: str = None, decimals: int = 18, chain_id: int = None):
        self.name = name
        self.api_url = api_url.rstrip('/')  # Ensure consistent URL format
        self.api_key = api_key
        self.decimals = decimals
        self.chain_id = chain_id

    def get_balance_url(self, address: str) -> str:
        """Generate properly formatted API URL for balance checks"""
        if self.name == "BTC":
            return f"{self.api_url}/{address}"
        return (f"{self.api_url}?module=account&action=balance"
                f"&address={address}&apikey={self.api_key}")

# Updated network configurations with current API endpoints
NETWORKS = {
    "ETH": Network("ETH", "https://api.etherscan.io/api", chain_id=1),
    "BTC": Network("BTC", "https://api.blockchair.com/bitcoin/dashboards/address", decimals=8),
    "MATIC": Network("MATIC", "https://api.polygonscan.com/api", chain_id=137),
    "ARB": Network("ARB", "https://api.arbiscan.io/api", chain_id=42161),
    "BSC": Network("BSC", "https://api.bscscan.com/api", chain_id=56),
    "BASE": Network("BASE", "https://api.basescan.org/api", chain_id=8453),
    "AVAX": Network("AVAX", "https://api.snowtrace.io/api", chain_id=43114),
    "FTM": Network("FTM", "https://api.ftmscan.com/api", chain_id=250),
    "OP": Network("OP", "https://api-optimistic.etherscan.io/api", chain_id=10),
    "CELO": Network("CELO", "https://api.celoscan.io/api", chain_id=42220),
    "CRONOS": Network("CRONOS", "https://api.cronoscan.com/api", chain_id=25),
    "GNOSIS": Network("GNOSIS", "https://api.gnosisscan.io/api", chain_id=100),
    "LINEA": Network("LINEA", "https://api.lineascan.build/api", chain_id=59144),
    "ZKSYNC": Network("ZKSYNC", "https://api.zksync.io/api", chain_id=324),
    "ALL": None
}

class ConfigManager:
    """Secure configuration management"""
    @staticmethod
    def load() -> Dict:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Config file corrupted: {e}")
                return {}
        return {}

    @staticmethod
    def save(config: Dict) -> None:
        CONFIG_FILE.parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Validate API key format"""
        return bool(key and 32 <= len(key) <= 64 and key.isalnum())

def setup_api_keys(force_setup: bool = False) -> Dict:
    """Interactive API key configuration"""
    config = ConfigManager.load()
    if not config or force_setup:
        print("\nAPI Key Setup")
        print("  API keys are required for each network. This tool uses Etherscan, Polygonscan, Blockchair")
        print("  and many other APIs. Please obtain API keys from the respective providers. See README for more details or ")
        print("  see the built in help dialogue by running 'python3 helsinki.py -h'")
        print("=" * 50)
        print("Enter API keys for each network (32-64 characters)")
        config = {}
        
        for name, network in NETWORKS.items():
            if name != 'ALL':
                while True:
                    key = input(f"\nEnter API key for {name} (press Enter to skip): ").strip()
                    if not key:
                        break
                    if ConfigManager.validate_api_key(key):
                        config[name] = key
                        break
                    print("Invalid API key format. Must be 32-64 alphanumeric characters.")
        
        ConfigManager.save(config)
        logger.info("API configuration updated successfully")
    
    return config

class WalletGenerator:
    """Secure wallet generation implementation"""
    @staticmethod
    def load_wordlist() -> List[str]:
        try:
            with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
                return [w.strip() for w in f.readlines()]
        except FileNotFoundError:
            logger.critical(f"Required file not found: {WORD_LIST_PATH}")
            raise SystemExit(1)
        except Exception as e:
            logger.critical(f"Failed to load wordlist: {e}")
            raise SystemExit(1)

    @staticmethod
    def generate_seed() -> bytes:
        """Generate cryptographically secure random seed"""
        return secrets.token_bytes(32)

    @staticmethod
    def generate_mnemonic(wordlist: List[str]) -> str:
        """Generate BIP39 compliant mnemonic"""
        try:
            entropy = WalletGenerator.generate_seed()
            h = hashlib.sha256(entropy).hexdigest()
            binary = bin(int.from_bytes(entropy, byteorder='big'))[2:].zfill(256)
            checksum = bin(int(h[0:2], 16))[2:].zfill(8)
            bits = binary + checksum
            
            return ' '.join(
                wordlist[int(bits[i:i+11], 2)]
                for i in range(0, len(bits)-8, 11)
            )
        except Exception as e:
            logger.error(f"Mnemonic generation failed: {e}")
            raise

    @staticmethod
    def derive_address(mnemonic: str) -> str:
        """Generate deterministic address from mnemonic"""
        try:
            seed = hashlib.sha256(mnemonic.encode()).digest()
            return '0x' + binascii.hexlify(seed[:20]).decode()
        except Exception as e:
            logger.error(f"Address derivation failed: {e}")
            raise

    @classmethod
    def generate_wallet(cls) -> Tuple[str, str]:
        """Generate complete wallet with mnemonic and address"""
        wordlist = cls.load_wordlist()
        mnemonic = cls.generate_mnemonic(wordlist)
        address = cls.derive_address(mnemonic)
        return mnemonic, address
class BalanceChecker:
    """Network balance checking implementation"""
    @staticmethod
    def check_balance(address: str, network: Network, retry_count: int = 0) -> float:
        try:
            headers = {
                'User-Agent': f'Helsinki/{VERSION}',
                'Accept': 'application/json'
            }
            
            url = network.get_balance_url(address)
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                data = json.loads(response.read())
            
            if network.name == "BTC":
                balance = data.get('data', {}).get(address, {}).get('balance', 0)
                return float(balance if balance is not None else 0) / (10 ** network.decimals)
            else:
                result = data.get('result', '0')
                return float(result if result is not None else 0) / (10 ** network.decimals)
            
        except urllib.error.HTTPError as e:
            if retry_count < MAX_RETRIES:
                logger.warning(f"Retrying {network.name} after HTTP error: {e.code}")
                time.sleep(1)
                return BalanceChecker.check_balance(address, network, retry_count + 1)
            logger.error(f"HTTP Error ({network.name}): {e.code}")
            return 0.0
        except Exception as e:
            logger.error(f"API Error ({network.name}): {str(e)}")
            return 0.0
        class ResultLogger:
            """Enhanced result logging with metrics"""
            @staticmethod
            def log_finding(mnemonic: str, address: str, balances: List[Dict], metrics: Dict) -> None:
                try:
                    LOGS_DIR.mkdir(exist_ok=True)
            
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = LOGS_DIR / f"found_{timestamp}.json"
            
                    data = {
                        "timestamp": timestamp,
                        "mnemonic": mnemonic,
                        "address": address,
                        "balances": balances,
                        "performance_metrics": {
                            "total_attempts": metrics['total_attempts'],
                            "success_rate": f"{metrics['success_rate']:.2%}",
                            "average_latency": f"{metrics['average_latency']:.3f}s",
                            "attempts_per_second": f"{metrics['attempts_per_second']:.2f}",
                            "memory_usage": metrics.get('memory_usage', {}),
                            "api_stats": {
                                "successful_calls": metrics.get('successful_calls', 0),
                                "failed_calls": metrics.get('failed_calls', 0)
                            }
                        }
                    }
            
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Results and metrics logged to {filename}")
                except Exception as e:
                    logger.error(f"Failed to log results: {str(e)}")
    @staticmethod
    def log_finding(mnemonic: str, address: str, balances: List[Dict], metrics: Dict) -> None:
        try:
            LOGS_DIR.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = LOGS_DIR / f"found_{timestamp}.json"
            
            data = {
                "timestamp": timestamp,
                "mnemonic": mnemonic,
                "address": address,
                "balances": balances,
                "performance_metrics": {
                    "total_attempts": metrics['total_attempts'],
                    "success_rate": f"{metrics['success_rate']:.2%}",
                    "average_latency": f"{metrics['average_latency']:.3f}s",
                    "attempts_per_second": f"{metrics['attempts_per_second']:.2f}",
                    "memory_usage": metrics.get('memory_usage', {}),
                    "api_stats": {
                        "successful_calls": metrics.get('successful_calls', 0),
                        "failed_calls": metrics.get('failed_calls', 0)
                    }
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Results and metrics logged to {filename}")
        except Exception as e:
            logger.error(f"Failed to log results: {str(e)}")

def get_active_networks(network_choice: str, config: Dict) -> List[Network]:
    """Get configured networks for checking"""
    if network_choice == 'ALL':
        return [
            Network(name, net.api_url, api_key=config.get(name), decimals=net.decimals, chain_id=net.chain_id)
            for name, net in NETWORKS.items()
            if name != 'ALL' and name in config
        ]
    if network_choice in config:
        net = NETWORKS[network_choice]
        return [Network(network_choice, net.api_url, api_key=config[network_choice], 
                       decimals=net.decimals, chain_id=net.chain_id)]
    return []

def parse_args():
                   """Command line argument parser"""
                   network_help = """
                   Supported Networks and Their API Providers:
    
                   ETH     - Etherscan.io API
                   BTC     - Blockchair.com API
                   MATIC   - Polygonscan.com API
                   ARB     - Arbiscan.io API
                   BSC     - BscScan.com API
                   BASE    - BaseScan.org API
                   AVAX    - Snowtrace.io API
                   FTM     - FTMScan.com API
                   OP      - Optimistic.Etherscan.io API
                   CELO    - CeloScan.io API
                   CRONOS  - CronosScan.com API
                   GNOSIS  - GnosisScan.io API
                   LINEA   - LineaScan.build API
                   ZKSYNC  - ZkSync.io API
                   ALL     - Check all configured networks
                   """
    
                   parser = argparse.ArgumentParser(
                       description=f"Helsinki Crypto Wallet Generator v{VERSION}",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       epilog=network_help
                   )
    
                   parser.add_argument('-v', '--verbose', 
                                      action='store_true',
                                      help='Show detailed running statistics')
                   parser.add_argument('-d', '--delay',
                                      type=float,
                                      default=API_DELAY,
                                      help='Delay between API calls in seconds')
                   parser.add_argument('-n', '--network',
                                      choices=list(NETWORKS.keys()),
                                      default='ALL',
                                      help='Target network for balance checking')
                   parser.add_argument('-a', '--attempts',
                                      type=int,
                                      help='Number of attempts (default: prompt for value)')
                   parser.add_argument('--setup', 
                                      action='store_true',
                                      help='Force API key setup')
    
                   return parser.parse_args()
DEFAULT_MAX_ATTEMPTS: int = 50

def get_max_attempts(args) -> int:
    if args.attempts is not None:
        return args.attempts
        
    config = ConfigManager.load()
    if 'max_attempts' in config:
        return config['max_attempts']
        
    # Return default value without prompting
    return DEFAULT_MAX_ATTEMPTS

def continuous_check(verbose: bool = False, 
                    delay: float = API_DELAY, 
                    network_choice: str = 'ALL',
                    max_attempts: int = 0,
                    active_networks: List[Network] = None) -> None:
    """Main processing loop"""
    global ATTEMPT_COUNTER
    start_time = time.time()
    metrics_collector = MetricsCollector()
    
    while True:
        try:
            ATTEMPT_COUNTER += 1
            if max_attempts and isinstance(max_attempts, int) and ATTEMPT_COUNTER > max_attempts:
                logger.info(f"Reached maximum attempts: {max_attempts}")
                break
                
            mnemonic, address = WalletGenerator.generate_wallet()
            balances = []
            found_balance = False
            
            if verbose:
                print(f"\nTesting phrase: {mnemonic}")
                print(f"Testing address: {address}")
            
            for network in active_networks or []:
                try:
                    balance = float(BalanceChecker.check_balance(address, network) or 0)
                    if balance > 0:
                        found_balance = True
                        balances.append({
                            "network": network.name,
                            "balance": balance
                        })
                        logger.info(f"Found balance! Network: {network.name}, Amount: {balance}")
                except (TypeError, ValueError):
                    continue
            
            if found_balance:
                ResultLogger.log_finding(
                    mnemonic, 
                    address, 
                    balances, 
                    metrics_collector.get_statistics()
                )
            
            elapsed_time = time.time() - start_time
            checks_per_second = ATTEMPT_COUNTER / elapsed_time
            
            status_msg = (f"\rAttempts: {ATTEMPT_COUNTER}/{max_attempts if max_attempts > 0 else 'inf'} | "
                         f"Rate: {checks_per_second:.2f} checks/s")
            
            if verbose:
                print(f"{status_msg}")
                print(f"Elapsed time: {elapsed_time:.2f}s")
                print("-" * 50)
            else:
                print(status_msg, end="", flush=True)
            
            time.sleep(delay)
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(1)
            continue
# Add to main() function at the start:
def main():
    """Main function"""
    try:
        initialize_environment()
        args = parse_args()
        config = setup_api_keys(force_setup=args.setup)
        
        if not config:
            logger.error("No API keys configured. Run with --setup to configure keys.")
            return
            
        active_networks = get_active_networks(args.network, config)
        if not active_networks:
            logger.error(f"No API key configured for network: {args.network}")
            return

        # Get max attempts with proper type handling
        max_attempts = get_max_attempts(args)
        if max_attempts is None:
            max_attempts = DEFAULT_MAX_ATTEMPTS

        logger.info(f"Helsinki v{VERSION} - Starting wallet generator")
        logger.info(f"Active Networks: {', '.join(net.name for net in active_networks)}")
        logger.info(f"Maximum attempts: {max_attempts}")
        print("Press Ctrl+C to stop...")
        
        continuous_check(
            verbose=args.verbose,
            delay=args.delay,
            network_choice=args.network,
            max_attempts=int(max_attempts),
            active_networks=active_networks
        )
    except KeyboardInterrupt:
        logger.info(f"Stopping... Final attempt count: {ATTEMPT_COUNTER}")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        raise
if __name__ == "__main__":
    main()

