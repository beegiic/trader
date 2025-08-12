#!/usr/bin/env python3
"""
Quick test script to verify Binance API connectivity
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment
load_dotenv()

def test_binance_api():
    """Test Binance API connection and basic calls."""
    
    print("🧪 Testing Binance API...")
    
    # Get credentials
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY") 
    use_testnet = os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true"
    use_futures = os.getenv("BINANCE_FUTURES", "true").lower() == "true"
    
    print(f"📊 Config:")
    print(f"  API Key: {api_key[:10]}..." if api_key else "  API Key: None")
    print(f"  Testnet: {use_testnet}")
    print(f"  Futures: {use_futures}")
    
    if not api_key or not secret_key:
        print("❌ Missing API credentials")
        return
    
    try:
        # Initialize client
        print("\n🔌 Initializing client...")
        client = Client(
            api_key=api_key,
            api_secret=secret_key,
            testnet=use_testnet
        )
        print("✅ Client initialized")
        
        # Test server time
        print("\n⏰ Testing server time...")
        server_time = client.get_server_time()
        print(f"✅ Server time: {server_time}")
        
        # Test account info
        print("\n👤 Testing account info...")
        if use_futures:
            account = client.futures_account()
            print(f"✅ Futures account balance: {account.get('totalWalletBalance', 0)}")
        else:
            account = client.get_account()
            usdt_balance = sum(
                float(asset['free']) + float(asset['locked']) 
                for asset in account['balances'] 
                if asset['asset'] == 'USDT'
            )
            print(f"✅ Spot USDT balance: {usdt_balance}")
        
        # Test ticker
        print("\n📈 Testing ticker data...")
        symbol = "BTCUSDT"
        
        if use_futures:
            ticker = client.futures_ticker(symbol=symbol)
        else:
            ticker = client.get_ticker(symbol=symbol)
            
        print(f"✅ {symbol} ticker:")
        print(f"  Price: ${float(ticker['lastPrice']):.2f}")
        print(f"  Change: {float(ticker['priceChangePercent']):.2f}%")
        print(f"  Volume: {float(ticker['volume']):.0f}")
        
        print("\n🎉 All tests passed!")
        
    except BinanceAPIException as e:
        print(f"❌ Binance API Error: {e}")
        print(f"   Code: {e.code}")
        print(f"   Message: {e.message}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binance_api()