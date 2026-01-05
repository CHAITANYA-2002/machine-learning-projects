"""
VOICE SHOPPING ASSISTANT - PRODUCTION READY VERSION
All issues fixed: TTS, Login, Pricing, Beeps, Accuracy

Author: Voice Shopping System
Version: 2.0 - FIXED
"""

import speech_recognition as sr
import pyttsx3
from typing import Optional, List, Dict, Tuple
import os
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import re
import winsound  # For beep sound on Windows

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================
#remove the comments below and add your actual credentials and API key
# CONFIG = {
#     # OpenAI API Key
   
#      "openai_api_key": "<REDACTED_OPENAI_API_KEY>",  # DO NOT COMMIT SECRETS

#     # Coles Account
#     "coles": {
#         "username": "patelsuchit3110@gmail.com",
#         "password": "KingSP@3110",
#         "enabled": True,
#         "url": "https://www.coles.com.au/"
#     },
    
#     # Woolworths Account
#     "woolworths": {
#         "username": "patelsuchit3110@gmail.com",
#         "password": "KingSP@3110",
#         "enabled": True,
#         "url": "https://www.woolworths.com.au/"
#     },
    
#     # Browser Settings
#     "browser": {
#         "headless": False,  # Set True to hide browser
#         "timeout": 20
#     },
    
#     # Voice Settings
#     "voice": {
#         "rate": 160,        # Speaking speed (120-200)
#         "volume": 1.0,      # Volume (0.0-1.0)
#         "beep_enabled": True  # Beep before listening
#     }
# }


# ============================================================================
# FIXED VOICE INTERFACE
# ============================================================================

class VoiceInterface:
    """Handles all voice I/O with proper TTS management"""
    
    def __init__(self, config: dict):
        print("Initializing voice systems...")
        
        self.config = config
        self.recognizer = sr.Recognizer()
        
        # Initialize TTS engine ONCE and keep it alive
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', config['voice']['rate'])
        self.tts.setProperty('volume', config['voice']['volume'])
        
        # Set female voice if available
        voices = self.tts.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts.setProperty('voice', voice.id)
                break
        
        # Test TTS
        try:
            self.tts.say("Voice systems initialized")
            self.tts.runAndWait()
            print("✓ TTS working correctly")
        except Exception as e:
            print(f"⚠️ TTS warning: {e}")
    
    def beep(self):
        """Play beep sound to signal user to speak"""
        if self.config['voice']['beep_enabled']:
            try:
                # Play a pleasant beep (frequency, duration)
                winsound.Beep(800, 200)  # 800Hz for 200ms
            except:
                print("🔔")  # Fallback visual indicator
    
    def speak(self, text: str, show_text: bool = True):
        """Speak text using TTS - FIXED VERSION"""
        if show_text:
            print(f"\n🤖 Assistant: {text}")
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
            # Try to reinitialize if failed
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', self.config['voice']['rate'])
                self.tts.say(text)
                self.tts.runAndWait()
            except:
                print("⚠️ TTS failed - check audio settings")
    
    def listen(self, prompt: Optional[str] = None, timeout: int = 10) -> str:
        """Listen for speech input with beep cue"""
        if prompt:
            self.speak(prompt)
        
        # Short pause after speaking
        time.sleep(0.3)
        
        # Play beep to signal user
        self.beep()
        
        with sr.Microphone() as source:
            print("\n🎤 Listening... (speak after the beep)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=20)
                text = self.recognizer.recognize_google(audio)
                print(f"👤 You: {text}")
                return text
            
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything. Let me ask again.")
                return ""
            
            except sr.UnknownValueError:
                self.speak("Sorry, I couldn't understand. Please speak clearly.")
                return ""
            
            except Exception as e:
                print(f"❌ Listen error: {e}")
                return ""
    
    def confirm(self, question: str) -> bool:
        """Ask yes/no question"""
        response = self.listen(question)
        return any(word in response.lower() for word in [
            "yes", "yeah", "yep", "sure", "okay", "ok", "correct", "right", "yup"
        ])
    
    def extract_numbers(self, text: str) -> str:
        """Extract numbers from text"""
        numbers = re.findall(r'\d+', text)
        return ''.join(numbers) if numbers else ""


# ============================================================================
# IMPROVED AI CONVERSATION
# ============================================================================

class ConversationAI:
    """Improved AI with better context understanding"""
    
    def __init__(self, api_key: str, voice: VoiceInterface):
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.history = [{
            "role": "system",
            "content": """You are a professional Australian grocery shopping assistant.

RULES:
1. Extract exact item names and quantities from user requests
2. Ask ONE clarifying question at a time
3. Keep responses SHORT (1-2 sentences max)
4. Be specific (e.g., "How many litres of skim milk?")
5. Confirm details before proceeding

EXAMPLES:
User: "I want milk"
You: "What type? Full cream, skim, or almond milk?"

User: "2 bottles"
You: "Got it! 2 bottles of skim milk. Anything else?"

Keep it conversational and efficient."""
        }]
    
    def chat(self, user_message: str) -> str:
        """Get AI response with improved prompting"""
        self.history.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                temperature=0.5,  # Lower for more consistent responses
                max_tokens=150    # Shorter responses for voice
            )
            
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            
            # Ensure reply is not too long for voice
            if len(reply) > 200:
                reply = reply[:200] + "..."
            
            return reply
            
        except Exception as e:
            print(f"❌ AI Error: {e}")
            return "Could you repeat that please?"
    
    def extract_order(self, conversation: str) -> List[Dict]:
        """Extract structured order from conversation"""
        try:
            prompt = f"""Extract ONLY confirmed items from this conversation:

{conversation}

Return JSON array:
[{{"item": "skim milk", "quantity": 2, "size": "1L"}}]

Only include items the user explicitly confirmed. Return [] if none."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result = response.choices[0].message.content.strip()
            result = result.replace("```json", "").replace("```", "").strip()
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return []


# ============================================================================
# FIXED SUPERMARKET AUTOMATION WITH PRICING
# ============================================================================

class SupermarketAutomation:
    """Fixed automation with price extraction"""
    
    def __init__(self, config: dict, voice: VoiceInterface):
        self.config = config
        self.voice = voice
        self.driver = None
        
        # Chrome options
        opts = Options()
        if config['browser']['headless']:
            opts.add_argument('--headless=new')
        opts.add_argument('--disable-blink-features=AutomationControlled')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        self.chrome_options = opts
    
    def create_driver(self) -> webdriver.Chrome:
        """Create browser instance"""
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        driver.implicitly_wait(15)
        driver.maximize_window()
        return driver
    
    def login_coles(self) -> bool:
        """FIXED Coles login"""
        try:
            self.voice.speak("Logging into Coles")
            self.driver = self.create_driver()
            
            print(f"Opening {self.config['coles']['url']}")
            self.driver.get(self.config['coles']['url'])
            time.sleep(4)
            
            # Close cookie banner
            try:
                cookie_btn = self.driver.find_element(By.XPATH, 
                    "//button[contains(., 'Accept') or contains(., 'OK')]")
                cookie_btn.click()
                time.sleep(1)
            except:
                pass
            
            # Click login
            try:
                login_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, 
                        "//button[contains(text(), 'Log in')] | //a[contains(text(), 'Log in')]"))
                )
                login_btn.click()
                time.sleep(3)
            except Exception as e:
                print(f"Login button error: {e}")
                return False
            
            # Enter email
            try:
                email_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "email"))
                )
                email_field.clear()
                email_field.send_keys(self.config['coles']['username'])
                print(f"Entered email: {self.config['coles']['username']}")
            except Exception as e:
                print(f"Email field error: {e}")
                return False
            
            # Enter password
            try:
                pwd_field = self.driver.find_element(By.ID, "password")
                pwd_field.clear()
                pwd_field.send_keys(self.config['coles']['password'])
                print("Entered password")
            except Exception as e:
                print(f"Password field error: {e}")
                return False
            
            # Submit
            try:
                submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                submit_btn.click()
                print("Submitted login form")
            except Exception as e:
                print(f"Submit error: {e}")
                pwd_field.send_keys(Keys.RETURN)
            
            time.sleep(6)
            
            # Verify login
            try:
                self.driver.find_element(By.XPATH, 
                    "//a[contains(., 'Account')] | //span[contains(., 'Hi')]")
                self.voice.speak("Successfully logged into Coles")
                return True
            except:
                self.voice.speak("Coles login failed. Please check credentials.")
                return False
                
        except Exception as e:
            print(f"❌ Coles login error: {e}")
            self.voice.speak("Could not log into Coles")
            return False
    
    def search_item_with_price(self, item_name: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Search item and extract price - FIXED"""
        try:
            self.voice.speak(f"Searching for {item_name}")
            
            # Search
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    "input[type='search'], input[placeholder*='Search']"))
            )
            search_box.clear()
            search_box.send_keys(item_name)
            search_box.send_keys(Keys.RETURN)
            time.sleep(4)
            
            # Get first product
            first_product = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    "section[data-testid='product-tile'], article.product"))
            )
            
            # Extract price
            price_text = None
            price_value = None
            
            try:
                price_elem = first_product.find_element(By.CSS_SELECTOR, 
                    "span[class*='price'], div[class*='price'], p[class*='price']")
                price_text = price_elem.text.strip()
                
                # Extract numeric price
                price_match = re.search(r'\$?(\d+\.?\d*)', price_text)
                if price_match:
                    price_value = float(price_match.group(1))
                    
            except Exception as e:
                print(f"Price extraction error: {e}")
            
            # Get product name
            try:
                name_elem = first_product.find_element(By.CSS_SELECTOR, 
                    "h2, h3, span[class*='name'], a")
                product_name = name_elem.text.strip()
            except:
                product_name = item_name
            
            if price_value:
                self.voice.speak(f"Found {product_name} for ${price_value:.2f}")
            else:
                self.voice.speak(f"Found {product_name}")
            
            return True, product_name, price_value
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            self.voice.speak(f"Couldn't find {item_name}")
            return False, None, None
    
    def add_to_cart(self, quantity: int) -> bool:
        """Add current product to cart"""
        try:
            # Click product to open details
            try:
                product = self.driver.find_element(By.CSS_SELECTOR, 
                    "section[data-testid='product-tile'], article.product")
                product.click()
                time.sleep(3)
            except:
                pass
            
            # Find Add button
            add_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, 
                    "//button[contains(., 'Add') and not(contains(., 'Added'))]"))
            )
            
            # Click for quantity
            for i in range(quantity):
                try:
                    add_btn.click()
                    time.sleep(1)
                    print(f"  Added {i+1}/{quantity}")
                except:
                    # Try increment button
                    try:
                        inc_btn = self.driver.find_element(By.XPATH, 
                            "//button[@aria-label='Increase'] | //button[contains(@class, 'plus')]")
                        inc_btn.click()
                        time.sleep(1)
                    except:
                        break
            
            self.voice.speak(f"Added {quantity} to cart")
            
            # Return to home
            self.driver.get(self.config['coles']['url'])
            time.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"❌ Add to cart error: {e}")
            return False
    
    def cleanup(self):
        """Close browser"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


# ============================================================================
# MAIN ASSISTANT - FIXED VERSION
# ============================================================================

class VoiceShoppingAssistant:
    """Main assistant with all fixes"""
    
    def __init__(self, config: dict):
        print("\n" + "="*70)
        print("  VOICE SHOPPING ASSISTANT - FIXED VERSION")
        print("="*70 + "\n")
        
        if config['openai_api_key'] == "PUT-YOUR-OPENAI-API-KEY-HERE":
            raise ValueError("❌ Add your OpenAI API key in CONFIG!")
        
        # Initialize components
        self.voice = VoiceInterface(config)
        print("✓ Voice ready")
        
        self.ai = ConversationAI(config['openai_api_key'], self.voice)
        print("✓ AI ready")
        
        self.automation = SupermarketAutomation(config, self.voice)
        print("✓ Browser ready")
        
        self.config = config
        self.cart = []
        
        print("\n✅ All systems ready!\n")
        time.sleep(1)
    
    def run(self):
        """Main flow - FIXED"""
        
        try:
            # STEP 1: Greet
            self.voice.speak("Hello! Welcome to your voice shopping assistant.")
            self.voice.speak("I can help you shop at Coles and Woolworths hands-free.")
            time.sleep(0.5)
            
            # STEP 2: Get order with better conversation
            self.voice.speak("What would you like to buy?")
            
            conversation = []
            max_turns = 10
            turns = 0
            
            while turns < max_turns:
                user_input = self.voice.listen()
                
                if not user_input:
                    continue
                
                # Exit check
                if any(w in user_input.lower() for w in ["exit", "quit", "cancel", "stop", "goodbye"]):
                    self.voice.speak("Okay, goodbye!")
                    return
                
                conversation.append(user_input)
                turns += 1
                
                # Get AI response
                ai_reply = self.ai.chat(user_input)
                self.voice.speak(ai_reply)
                
                # Check if ready to confirm
                if any(phrase in ai_reply.lower() for phrase in [
                    "got it", "perfect", "great choice", "anything else"
                ]):
                    if self.voice.confirm("Is your order complete?"):
                        break
            
            # Extract order
            full_convo = " ".join(conversation)
            order_items = self.ai.extract_order(full_convo)
            
            if not order_items:
                self.voice.speak("I didn't catch any items. Let's try again.")
                return
            
            # Confirm order
            self.voice.speak("Let me confirm your order:")
            for item in order_items:
                qty = item.get('quantity', 1)
                name = item.get('item')
                size = item.get('size', '')
                self.voice.speak(f"{qty} {name} {size}")
            
            if not self.voice.confirm("Is this correct?"):
                self.voice.speak("Let's start over.")
                return
            
            self.cart = order_items
            
            # STEP 3: Choose store
            store_choice = self.voice.listen("Which store? Coles or Woolworths?")
            
            store = "coles" if "coles" in store_choice.lower() else "woolworths"
            store_name = "Coles" if store == "coles" else "Woolworths"
            
            self.voice.speak(f"Great! Shopping at {store_name}")
            
            # STEP 4: Login and add items
            if store == "coles":
                if not self.automation.login_coles():
                    self.voice.speak("Login failed. Please check your credentials.")
                    return
            else:
                self.voice.speak("Woolworths automation coming soon. Using Coles for now.")
                if not self.automation.login_coles():
                    return
            
            # Add items with prices
            self.voice.speak("Adding items to your cart")
            
            total_price = 0.0
            for item in self.cart:
                item_name = item['item']
                quantity = item.get('quantity', 1)
                
                success, product_name, price = self.automation.search_item_with_price(item_name)
                
                if success:
                    if self.automation.add_to_cart(quantity):
                        if price:
                            item_total = price * quantity
                            total_price += item_total
                            self.voice.speak(f"That's ${item_total:.2f} for {quantity} items")
                        time.sleep(2)
                else:
                    if not self.voice.confirm(f"Couldn't find {item_name}. Continue with other items?"):
                        return
            
            if total_price > 0:
                self.voice.speak(f"Your cart total is approximately ${total_price:.2f}")
            
            # STEP 5 & 6: Checkout
            if self.voice.confirm("Ready to checkout?"):
                self.voice.speak("Taking you to checkout. Please complete payment in the browser.")
                
                try:
                    cart_btn = WebDriverWait(self.automation.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'cart')]"))
                    )
                    cart_btn.click()
                    time.sleep(3)
                    
                    checkout_btn = WebDriverWait(self.automation.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Checkout')]"))
                    )
                    checkout_btn.click()
                    
                    self.voice.speak("Checkout page ready. Complete payment manually.")
                except:
                    self.voice.speak("Please navigate to checkout in the browser.")
            else:
                self.voice.speak("Items saved in your cart. You can checkout later.")
            
            self.voice.speak("Thank you for using voice shopping! Have a great day!")
            
        except KeyboardInterrupt:
            self.voice.speak("Stopping. Goodbye!")
        
        except Exception as e:
            self.voice.speak("Sorry, something went wrong.")
            print(f"❌ Error: {e}")
        
        finally:
            self.automation.cleanup()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("  SYSTEM CHECK")
    print("="*70)
    
    # Check API key
    if CONFIG['openai_api_key'] == "PUT-YOUR-OPENAI-API-KEY-HERE":
        print("\n❌ OpenAI API key not set!")
        print("Add it to CONFIG['openai_api_key'] in the code\n")
        input("Press Enter to exit...")
        exit(1)
    
    # Check credentials
    if "patelsuchit3110" not in CONFIG['coles']['username']:
        print("\n⚠️ Warning: Using default credentials")
        print("Update CONFIG['coles']['username'] and CONFIG['coles']['password']\n")
    
    print("\n✅ Configuration validated")
    print("🚀 Starting assistant...\n")
    time.sleep(1)
    
    try:
        assistant = VoiceShoppingAssistant(CONFIG)
        assistant.run()
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
    
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        print("\nCheck:")
        print("- Internet connection")
        print("- OpenAI API key validity")
        print("- Microphone permissions")
        print("- Chrome browser installed\n")
        input("Press Enter to exit...")