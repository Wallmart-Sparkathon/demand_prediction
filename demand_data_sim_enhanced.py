import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import requests, time
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
products = [
    ("Mithai Gift Box (Diwali)", "Festive", "#mithai #diwali #sweets #gifting #festivalvibes #celebration #tradition"),
    ("Rakhi Gift Hamper", "Festive", "#rakhi #giftbox #festival #siblings #love #tradition #celebration"),
    ("Holi Color Pack", "Festive", "#holi #colors #festivalofcolors #fun #celebration #spring #india"),
    ("Eid Sweet Box", "Festive", "#eid #sweets #celebration #iftar #festival #ramadan #gift"),
    ("Christmas Cake", "Festive", "#christmas #cake #holidayseason #baking #gifts #winter #joy"),
    ("Navratri Fasting Food Pack", "Festive", "#navratri #fasting #indianfestivals #spiritual #food #cleanse #devotion"),
    ("Firecrackers Pack", "Festive", "#firecrackers #diwali #celebration #lights #sound #festival #tradition"),
    ("New Year Party Kit", "Events", "#newyear #party #celebration #2025 #cheers #countdown #midnight"),
    ("Valentine's Day Chocolates", "Festive", "#valentine #chocolates #gift #love #romance #hearts #sweet"),
    ("Ganesh Chaturthi Modak Box", "Festive", "#ganeshchaturthi #modak #sweets #devotion #celebration #traditional"),
    ("IPL Team Jersey", "Sports", "#ipl #jersey #cricket #teamspirit #matchday #sportswear #fans"),
    ("World Cup Cricket Kit", "Sports", "#worldcup #cricket #kit #sportsgear #tournament #fans #teamindia"),
    ("National Flag Pack", "Events", "#flag #independenceday #republicday #patriotism #india #nationalpride"),
    ("Match Day Snack Box", "Events", "#snacks #matchday #cricket #watchparty #yum #quickbites #sports"),
    ("Bluetooth Speaker", "Electronics", "#speaker #bluetooth #music #party #portable #sound #bass"),
    ("Gaming Headset", "Electronics", "#gaming #headset #esports #soundquality #gamerlife #mic #pcgaming"),
    ("Live Streaming Dongle", "Electronics", "#streaming #dongle #chromecast #tv #entertainment #wifi #online"),
    ("Instant Popcorn Tub", "Snacks", "#popcorn #instantfood #movie #snacktime #microwave #easyfood"),
    ("Limited Edition Movie Merchandise", "Entertainment", "#movie #merch #collectible #fan #limitededition #exclusive #fandom"),
    ("Celebrity-endorsed Fragrance", "Personal Care", "#fragrance #celebrity #perfume #style #luxury #scent #signature"),
    ("Sweater (Winterwear)", "Apparel", "#sweater #winterwear #warm #cozy #style #fashion #coldweather"),
    ("Raincoat / Windcheater", "Apparel", "#raincoat #windcheater #monsoon #staydry #rainyseason #weatherproof"),
    ("Sunscreen SPF 50", "Personal Care", "#sunscreen #spf50 #summer #uvprotection #skincare #sunblock"),
    ("Sunglasses (UV Protect)", "Accessories", "#sunglasses #uvprotection #fashion #summerstyle #accessories #cool"),
    ("Cooling Gel Sheet", "Health", "#coolinggel #fever #relief #hotweather #comfort #instantrelief"),
    ("Herbal Tea for Cold", "Health", "#herbaltea #cold #remedy #wellness #healing #naturalcure"),
    ("Watermelon (Summer Fruit)", "Fruits", "#watermelon #summerfruit #refreshing #hydration #healthy #cooling"),
    ("Mosquito Repellent Spray", "Household", "#mosquito #repellent #spray #protection #health #insectcontrol"),
    ("Humidifier", "Household", "#humidifier #airquality #home #wintercare #moisture #dryskin"),
    ("Air Purifier", "Electronics", "#airpurifier #pollution #cleanair #health #home #freshair"),
    ("Hand Sanitizer", "Hygiene", "#sanitizer #hygiene #covid #cleanhands #germfree #protection"),
    ("N95 Mask Pack", "Hygiene", "#n95 #mask #protection #healthsafety #pandemic #facemask"),
    ("Immunity Booster Juice", "Health", "#immunity #juice #health #vitamins #wellness #boost"),
    ("Vitamin C Tablets", "Health", "#vitaminc #supplement #health #boost #immunity #antioxidant"),
    ("Herbal Kadha Kit", "Health", "#kadha #ayurveda #immunity #wellbeing #naturalremedy #tradition"),
    ("Digital Thermometer", "Health", "#thermometer #digital #healthmonitor #checkup #essential #tempcheck"),
    ("Oximeter", "Health", "#oximeter #oxygenlevel #monitor #covidcare #device #breathmonitor"),
    ("Steam Vaporizer", "Health", "#vaporizer #steam #coldrelief #nasalcare #wellness #respiratory"),
    ("Pain Relief Balm", "Health", "#painrelief #balm #musclerelief #natural #instantrelief #soothing"),
    ("Ayurvedic Chyawanprash", "Health", "#chyawanprash #ayurveda #immunity #healthtonic #tradition #herbal"),
    ("Korean Ramen Pack", "Snacks", "#koreanramen #noodles #spicy #quickmeal #asianfood #comfortfood"),
    ("Matcha Green Tea", "Beverages", "#matcha #greentea #superfood #antioxidants #energy #detox"),
    ("Dalgona Coffee Kit", "Beverages", "#dalgona #coffee #trending #diy #beverage #frothy"),
    ("Protein Pancake Mix", "Snacks", "#protein #pancake #fitness #breakfast #healthyeating #energyfood"),
    ("Hair Biotin Gummies", "Personal Care", "#biotin #haircare #gummies #growth #supplement #hairgrowth"),
    ("TikTok Viral Gadget", "Electronics", "#tiktok #viral #gadget #trending #tech #musthave"),
    ("Glow Serum (K-Beauty)", "Personal Care", "#kbeauty #glowserum #skincare #radiant #beautyroutine #glassskin"),
    ("Portable Blender Bottle", "Kitchen", "#blenderbottle #portable #fitness #smoothies #kitchen #shakes"),
    ("Mini Air Fryer", "Kitchen", "#airfryer #cooking #healthy #snacks #kitchenappliance #oilfree"),
    ("Anime Stationery Set", "Accessories", "#anime #stationery #otaku #cute #writing #kawaii")
]

# Enhanced product triggers with realistic demand patterns
product_triggers = {
    "Mithai Gift Box (Diwali)": {"festival": "Diwali"},
    "Rakhi Gift Hamper": {"festival": "Raksha Bandhan"},
    "Holi Color Pack": {"festival": "Holi"},
    "Eid Sweet Box": {"festival": "Eid"},
    "Christmas Cake": {"festival": "Christmas"},
    "Navratri Fasting Food Pack": {"festival": "Navratri"},
    "Firecrackers Pack": {"festival": "Diwali"},
    "New Year Party Kit": {"festival": "New Year"},
    "Valentine's Day Chocolates": {"festival": "Valentine's Day"},
    "Ganesh Chaturthi Modak Box": {"festival": "Ganesh Chaturthi"},
    "IPL Team Jersey": {"event_weeks": list(range(15, 23))},
    "World Cup Cricket Kit": {"event_weeks": list(range(36, 45))},
    "National Flag Pack": {"festival": "Independence Day"},
    "Match Day Snack Box": {"event_weeks": list(range(15, 23)) + list(range(36, 45))},
    "Bluetooth Speaker": {"festival": "Diwali"},
    "Gaming Headset": {"trend_weeks": list(range(20, 30))},
    "Live Streaming Dongle": {"event_weeks": list(range(15, 23)) + list(range(36, 45))},
    "Instant Popcorn Tub": {"event_weeks": list(range(15, 23)) + list(range(36, 45))},
    "Limited Edition Movie Merchandise": {"trend_weeks": list(range(1, 10))},
    "Celebrity-endorsed Fragrance": {"trend_weeks": list(range(25, 35))},
    "Sweater (Winterwear)": {"season": "Cold"},
    "Raincoat / Windcheater": {"season": "Rainy"},
    "Sunscreen SPF 50": {"season": "Hot"},
    "Sunglasses (UV Protect)": {"season": "Hot"},
    "Cooling Gel Sheet": {"season": "Hot"},
    "Herbal Tea for Cold": {"season": "Cold"},
    "Watermelon (Summer Fruit)": {"season": "Hot"},
    "Mosquito Repellent Spray": {"season": "Rainy"},
    "Humidifier": {"season": "Cold"},
    "Air Purifier": {"season": "Cold"},
    "Hand Sanitizer": {"trend_weeks": list(range(1, 52))},
    "N95 Mask Pack": {"trend_weeks": list(range(1, 52))},
    "Immunity Booster Juice": {"season": "Cold"},
    "Vitamin C Tablets": {"season": "Cold"},
    "Herbal Kadha Kit": {"season": "Cold"},
    "Digital Thermometer": {"season": "Cold"},
    "Oximeter": {"trend_weeks": list(range(1, 52))},
    "Steam Vaporizer": {"season": "Cold"},
    "Pain Relief Balm": {"season": "Cold"},
    "Ayurvedic Chyawanprash": {"season": "Cold"},
    "Korean Ramen Pack": {"trend_weeks": list(range(10, 20))},
    "Matcha Green Tea": {"trend_weeks": list(range(5, 15))},
    "Dalgona Coffee Kit": {"trend_weeks": list(range(30, 40))},
    "Protein Pancake Mix": {"trend_weeks": list(range(1, 10))},
    "Hair Biotin Gummies": {"trend_weeks": list(range(25, 35))},
    "TikTok Viral Gadget": {"trend_weeks": list(range(15, 25))},
    "Glow Serum (K-Beauty)": {"trend_weeks": list(range(20, 30))},
    "Portable Blender Bottle": {"trend_weeks": list(range(1, 10))},
    "Mini Air Fryer": {"trend_weeks": list(range(35, 45))},
    "Anime Stationery Set": {"trend_weeks": list(range(40, 50))}
}

festivals = {
    "Diwali": 44,
    "Raksha Bandhan": 32,
    "Holi": 10,
    "Eid": 23,
    "Christmas": 52,
    "Navratri": 40,
    "Ganesh Chaturthi": 36,
    "Independence Day": 33,
    "Republic Day": 4,
    "New Year": 1,
    "Valentine's Day": 7,
    "Onam": 35
}

weather_by_week = {
    "Hot": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "Rainy": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
    "Cold": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 1, 2, 3, 4, 5]
}

cities = ["Delhi", "Mumbai", "Bangalore"]
weeks = 52
start_date = datetime(2024, 1, 1)

# ---------------- ENHANCED DEMAND PATTERN FUNCTIONS ----------------

def get_base_demand(product, category):
    """Get base demand based on product category and realistic market patterns"""
    category_base = {
        "Festive": 85,
        "Events": 65,
        "Sports": 105,
        "Electronics": 125,
        "Snacks": 155,
        "Entertainment": 75,
        "Personal Care": 95,
        "Apparel": 115,
        "Accessories": 90,
        "Health": 100,
        "Fruits": 205,
        "Household": 80,
        "Hygiene": 135,
        "Beverages": 145,
        "Kitchen": 70
    }
    return category_base.get(category, 100)

def get_seasonal_demand_pattern(week):
    """Create seasonal demand patterns that gradually change"""
    seasonal_patterns = {
        "Hot": {
            "peak_weeks": [12, 13, 14, 15, 16, 17, 18],
            "gradual_weeks": [10, 11, 19, 20, 21],
            "multiplier": 2.8
        },
        "Rainy": {
            "peak_weeks": [25, 26, 27, 28, 29, 30, 31, 32, 33],
            "gradual_weeks": [23, 24, 34, 35],
            "multiplier": 2.3
        },
        "Cold": {
            "peak_weeks": [42, 43, 44, 45, 46, 47, 48, 49, 50],
            "gradual_weeks": [40, 41, 51, 52, 1, 2, 3],
            "multiplier": 2.5
        }
    }
    
    for season, pattern in seasonal_patterns.items():
        if week in pattern["peak_weeks"]:
            return season, pattern["multiplier"]
        elif week in pattern["gradual_weeks"]:
            return season, pattern["multiplier"] * 0.6
    
    return "Normal", 1.0

def get_festival_demand_curve(product, week):
    """Create realistic festival demand curves with pre and post festival effects"""
    triggers = product_triggers.get(product, {})
    
    if 'festival' in triggers:
        festival_week = festivals.get(triggers['festival'])
        if festival_week:
            week_diff = week - festival_week
            
            # Pre-festival buildup
            if week_diff == -3:
                return 1.5, "Pre-festival early"
            elif week_diff == -2:
                return 2.2, "Pre-festival medium"
            elif week_diff == -1:
                return 3.5, "Pre-festival peak"
            elif week_diff == 0:
                return 4.8, "Festival day"
            elif week_diff == 1:
                return 1.8, "Post-festival clearance"
            elif week_diff == 2:
                return 0.8, "Post-festival low"
            elif week_diff in [-4, -5]:
                return 1.2, "Early preparation"
                
    return 1.0, "Normal"

def get_trend_demand_pattern(product, week):
    """Create trending product demand patterns"""
    triggers = product_triggers.get(product, {})
    
    if 'trend_weeks' in triggers:
        trend_weeks = triggers['trend_weeks']
        if week in trend_weeks:
            # Position in trend cycle
            trend_position = (week - min(trend_weeks)) / len(trend_weeks)
            
            if trend_position <= 0.2:  # Early adoption
                return 1.8, "Trend emerging"
            elif trend_position <= 0.5:  # Growth phase
                return 2.8, "Trend growing"
            elif trend_position <= 0.8:  # Peak phase
                return 3.2, "Trend peak"
            else:  # Decline phase
                return 2.0, "Trend declining"
    
    return 1.0, "Normal"

def get_event_demand_pattern(product, week):
    """Create event-based demand patterns (IPL, World Cup, etc.)"""
    triggers = product_triggers.get(product, {})
    
    if 'event_weeks' in triggers:
        event_weeks = triggers['event_weeks']
        if week in event_weeks:
            # Different products peak at different times during events
            event_position = (week - min(event_weeks)) / len(event_weeks)
            
            # Sports merchandise peaks early
            if product in ["IPL Team Jersey", "World Cup Cricket Kit"]:
                if event_position <= 0.3:
                    return 3.5, "Event merchandise peak"
                elif event_position <= 0.7:
                    return 2.8, "Event merchandise sustained"
                else:
                    return 2.2, "Event merchandise decline"
            
            # Food/Entertainment items peak during mid-event
            elif product in ["Match Day Snack Box", "Instant Popcorn Tub", "Live Streaming Dongle"]:
                if event_position <= 0.2:
                    return 2.0, "Event food early"
                elif event_position <= 0.8:
                    return 3.2, "Event food peak"
                else:
                    return 2.5, "Event food late"
    
    return 1.0, "Normal"

def calculate_city_preference_multiplier(product, city):
    """Calculate city-specific preferences for products"""
    city_preferences = {
        "Delhi": {
            "Sweater (Winterwear)": 1.5,
            "Herbal Tea for Cold": 1.4,
            "Air Purifier": 1.6,
            "Humidifier": 1.3,
            "Gaming Headset": 1.2,
            "Firecrackers Pack": 1.4,
            "Mithai Gift Box (Diwali)": 1.3
        },
        "Mumbai": {
            "Raincoat / Windcheater": 1.6,
            "Mosquito Repellent Spray": 1.5,
            "IPL Team Jersey": 1.4,
            "Match Day Snack Box": 1.3,
            "Celebrity-endorsed Fragrance": 1.4,
            "Korean Ramen Pack": 1.3,
            "Bluetooth Speaker": 1.2
        },
        "Bangalore": {
            "Gaming Headset": 1.8,
            "TikTok Viral Gadget": 1.6,
            "Glow Serum (K-Beauty)": 1.5,
            "Matcha Green Tea": 1.4,
            "Protein Pancake Mix": 1.3,
            "Mini Air Fryer": 1.4,
            "Portable Blender Bottle": 1.3
        }
    }
    
    return city_preferences.get(city, {}).get(product, 1.0)

def get_weather(week):
    """Get weather for a specific week"""
    for weather, weeks_list in weather_by_week.items():
        if week in weeks_list:
            return weather
    return "Normal"

def get_trend_score(product, week):
    """Get trend score for a product"""
    # Simulate trend scores based on product type and week
    if "K-Beauty" in product or "TikTok" in product:
        return random.randint(40, 100)
    elif "Korean" in product or "Matcha" in product:
        return random.randint(30, 80)
    elif "Gaming" in product or "Bluetooth" in product:
        return random.randint(20, 70)
    else:
        return random.randint(10, 50)

def media_surge_flag(product):
    """Determine if product has media surge"""
    surge_products = ["Celebrity-endorsed Fragrance", "TikTok Viral Gadget", "Limited Edition Movie Merchandise"]
    return 1 if product in surge_products else 0

def generate_realistic_demand_quantities(product, category, week, city, weather):
    """Generate realistic demand quantities based on multiple sophisticated factors"""
    
    # Base demand from product category
    base_demand = get_base_demand(product, category)
    
    # Get seasonal multiplier
    season_type, seasonal_multiplier = get_seasonal_demand_pattern(week)
    
    # Get festival demand curve
    festival_multiplier, festival_phase = get_festival_demand_curve(product, week)
    
    # Get trend demand pattern
    trend_multiplier, trend_phase = get_trend_demand_pattern(product, week)
    
    # Get event demand pattern
    event_multiplier, event_phase = get_event_demand_pattern(product, week)
    
    # City preference multiplier
    city_multiplier = calculate_city_preference_multiplier(product, city)
    
    # Weather-based adjustments
    weather_multiplier = 1.0
    triggers = product_triggers.get(product, {})
    if 'season' in triggers:
        if weather == triggers['season']:
            weather_multiplier = 2.2
        elif weather == "Normal":
            weather_multiplier = 1.1
    
    # Calculate final demand
    total_multiplier = (seasonal_multiplier * festival_multiplier * trend_multiplier * 
                       event_multiplier * city_multiplier * weather_multiplier)
    
    # Base expected demand
    expected_demand = int(base_demand * total_multiplier)
    
    # Promotion logic - more likely during high demand periods
    promotion_probability = min(0.8, 0.2 + (total_multiplier - 1.0) * 0.3)
    promotion = 1 if np.random.random() < promotion_probability else 0
    
    # Add promotion boost
    if promotion:
        expected_demand = int(expected_demand * 1.4)
    
    # Add controlled variance (±10% instead of ±30%)
    variance = max(5, expected_demand * 0.1)
    outward_qty = max(1, int(np.random.normal(expected_demand, variance)))
    
    # Realistic stock management
    if total_multiplier > 2.0:  # High demand period
        stock_multiplier = 1.6
        restock_multiplier = 1.4
    elif total_multiplier > 1.5:  # Medium demand period
        stock_multiplier = 1.3
        restock_multiplier = 1.2
    else:  # Normal demand period
        stock_multiplier = 1.1
        restock_multiplier = 1.0
    
    # Current stock should be realistic
    current_stock = max(outward_qty, int(outward_qty * stock_multiplier + np.random.normal(0, 5)))
    
    # Inward quantity for restocking
    inward_qty = max(0, int(outward_qty * restock_multiplier + np.random.normal(0, 8)))
    
    return {
        'outward_qty': outward_qty,
        'inward_qty': inward_qty,
        'current_stock': current_stock,
        'promotion': promotion,
        'demand_multiplier': round(total_multiplier, 2),
        'festival_phase': festival_phase,
        'trend_phase': trend_phase,
        'event_phase': event_phase,
        'city_multiplier': round(city_multiplier, 2)
    }

# ---------------- MAIN SIMULATION ----------------

print("Starting Enhanced Demand Data Simulation...")
print(f"Total Products: {len(products)}")
print(f"Total Cities: {len(cities)}")
print(f"Total Weeks: {weeks}")
print(f"Total Records to Generate: {len(products) * len(cities) * weeks}")

np.random.seed(42)  # For reproducibility
random.seed(42)

data = []
for week in tqdm(range(1, weeks + 1), desc="Simulating Weeks"):
    date = start_date + timedelta(weeks=week - 1)
    weather = get_weather(week)
    holiday = 1 if week in festivals.values() else 0

    for city in cities:
        for product, category, hashtags in products:
            # Generate realistic demand quantities
            demand_data = generate_realistic_demand_quantities(product, category, week, city, weather)
            
            # Get trend and media data
            trend_score = get_trend_score(product, week)
            media_flag = media_surge_flag(product)
            
            data.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Week": week,
                "Product Name": product,
                "Category": category,
                "Store Location": city,
                "Outward Qty": demand_data['outward_qty'],
                "Inward Qty": demand_data['inward_qty'],
                "Current Stock": demand_data['current_stock'],
                "Promotion": demand_data['promotion'],
                "Holiday": holiday,
                "Weather": weather,
                "TrendScore": trend_score,
                "MediaSurge": media_flag,
                "Month": date.month,
                "Year": date.year,
                "Hashtags": hashtags,
                "DemandMultiplier": demand_data['demand_multiplier'],
                "FestivalPhase": demand_data['festival_phase'],
                "TrendPhase": demand_data['trend_phase'],
                "EventPhase": demand_data['event_phase'],
                "CityMultiplier": demand_data['city_multiplier']
            })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV and Excel
df.to_csv("enhanced_demand_dataset.csv", index=False)
df.to_excel("enhanced_demand_dataset.xlsx", index=False)

print(f"\n✅ Enhanced Dataset Generated Successfully!")
print(f"📊 Total Records: {len(df)}")
print(f"📁 Files Created:")
print(f"   - enhanced_demand_dataset.csv")
print(f"   - enhanced_demand_dataset.xlsx")

# Display sample statistics
print(f"\n📈 Sample Statistics:")
print(f"Average Outward Qty: {df['Outward Qty'].mean():.2f}")
print(f"Max Outward Qty: {df['Outward Qty'].max()}")
print(f"Min Outward Qty: {df['Outward Qty'].min()}")
print(f"Average Demand Multiplier: {df['DemandMultiplier'].mean():.2f}")
print(f"Max Demand Multiplier: {df['DemandMultiplier'].max():.2f}")

# Show top performing products by average demand
print(f"\n🏆 Top 10 Products by Average Demand:")
top_products = df.groupby('Product Name')['Outward Qty'].mean().sort_values(ascending=False).head(10)
for product, avg_demand in top_products.items():
    print(f"   {product}: {avg_demand:.2f}")

# Show seasonal patterns
print(f"\n🌦️ Seasonal Demand Patterns:")
seasonal_demand = df.groupby('Weather')['Outward Qty'].mean().sort_values(ascending=False)
for season, avg_demand in seasonal_demand.items():
    print(f"   {season}: {avg_demand:.2f}")

print(f"\n🎯 Dataset Features:")
print(f"   - Realistic festival demand curves with pre/post effects")
print(f"   - Seasonal demand patterns with gradual transitions")
print(f"   - Event-based demand (IPL, World Cup) with different product phases")
print(f"   - City-specific preferences for products")
print(f"   - Trend-based demand patterns")
print(f"   - Smart promotion logic based on demand multipliers")
print(f"   - Realistic stock management")
print(f"   - Weather-based demand adjustments")
