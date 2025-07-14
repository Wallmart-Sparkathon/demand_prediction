import requests
import json
import time

# Base URL for the Flask app
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        result = response.json()
        print(f"✅ Health Check: {result['status']}")
        print(f"   Model Loaded: {result['model_loaded']}")
        return result['model_loaded']
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        result = response.json()
        if result['success']:
            print(f"✅ Model Info Retrieved:")
            print(f"   Training R²: {result['training_r2']}")
            print(f"   Test R²: {result['test_r2']}")
            print(f"   Test MAE: {result['test_mae']}")
            print(f"   Features: {result['num_features']}")
            print(f"   Parameters: {result['model_parameters']:,}")
        else:
            print(f"❌ Model Info Failed: {result['error']}")
        return result['success']
    except Exception as e:
        print(f"❌ Model Info Failed: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🔍 Testing Single Prediction...")
    
    # Test data for a festive product during Diwali season
    test_data = {
        "Week": 40,
        "Month": 10,
        "Product Name": "Mithai Gift Box (Diwali)",
        "Category": "Festive",
        "Store Location": "Delhi",
        "Promotion": 1,
        "Holiday": 1,
        "TrendScore": 50,
        "DemandMultiplier": 2.0,
        "CityMultiplier": 1.3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(test_data)
        )
        result = response.json()
        
        if result['success']:
            print(f"✅ Prediction Successful:")
            print(f"   Product: {test_data['Product Name']}")
            print(f"   Store: {test_data['Store Location']}")
            print(f"   Predicted Demand: {result['prediction']} units")
            return True
        else:
            print(f"❌ Prediction Failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction Failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing Batch Prediction...")
    
    # Test data for multiple products
    test_data = {
        "inputs": [
            {
                "Week": 40,
                "Month": 10,
                "Product Name": "Mithai Gift Box (Diwali)",
                "Category": "Festive",
                "Store Location": "Delhi",
                "Promotion": 1,
                "Holiday": 1,
                "TrendScore": 50,
                "DemandMultiplier": 2.0,
                "CityMultiplier": 1.3
            },
            {
                "Week": 20,
                "Month": 5,
                "Product Name": "Gaming Headset",
                "Category": "Electronics",
                "Store Location": "Mumbai",
                "Promotion": 0,
                "Holiday": 0,
                "TrendScore": 35,
                "DemandMultiplier": 1.8,
                "CityMultiplier": 1.2
            },
            {
                "Week": 10,
                "Month": 3,
                "Product Name": "Hand Sanitizer",
                "Category": "Hygiene",
                "Store Location": "Bangalore",
                "Promotion": 1,
                "Holiday": 0,
                "TrendScore": 40,
                "DemandMultiplier": 2.5,
                "CityMultiplier": 1.1
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(test_data)
        )
        result = response.json()
        
        if result['success']:
            print(f"✅ Batch Prediction Successful:")
            print(f"   Total Processed: {result['total_processed']}")
            print(f"   Successful: {result['successful_predictions']}")
            print(f"   Failed: {result['failed_predictions']}")
            
            for pred in result['predictions']:
                product_name = pred['input']['Product Name']
                prediction = pred['prediction']
                print(f"   • {product_name}: {prediction} units")
                
            return True
        else:
            print(f"❌ Batch Prediction Failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Batch Prediction Failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🔍 Testing Error Handling...")
    
    # Test with missing required field
    invalid_data = {
        "Week": 40,
        "Month": 10,
        # Missing Product Name
        "Category": "Festive",
        "Store Location": "Delhi"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(invalid_data)
        )
        result = response.json()
        
        if not result['success'] and 'error' in result:
            print(f"✅ Error Handling Works:")
            print(f"   Error Message: {result['error']}")
            return True
        else:
            print(f"❌ Error Handling Failed: Should have returned error")
            return False
            
    except Exception as e:
        print(f"❌ Error Handling Test Failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 DEMAND PREDICTION API TESTS")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} Exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above.")
    
    print("\n💡 To test the web interface, open: http://localhost:5000")

if __name__ == "__main__":
    main()
