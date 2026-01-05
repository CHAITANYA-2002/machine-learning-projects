"""
VOICE SHOPPING ASSISTANT - FINAL VERSION
Complete voice-controlled shopping with intelligent conversation

FLOW:
1. Greet user
2. Ask for order + understand specific needs and quantities
3. Ask for store preference
4. Add items to cart
5. Ask about delivery options on screen
6. Complete payment with voice-provided details

Author: Voice Shopping System
Version: 1.0 Final
"""

import speech_recognition as sr
import pyttsx3
from typing import Optional, List, Dict
import os
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import re

# ============================================================================
# CONFIGURATION - ADD YOUR OPENAI API KEY
# ============================================================================

CONFIG = {
    "openai_api_key": "PUT-YOUR-OPENAI-API-KEY-HERE",  # ← ADD YOUR KEY HERE
    
    "coles": {
        "username": "patelsuchit3110@gmail.com",
        "password": "KingSP@3110",
        "enabled": True,
        "url": "https://www.coles.com.au/"
    },
    
    "woolworths": {
        "username": "patelsuchit3110@gmail.com",
        "password": "KingSP@3110",
        "enabled": True,
        "url": "https://www.woolworths.com.au/"
    },
    
    "browser": {
        "headless": False,  # Set True to hide browser
        "timeout": 20
    },
    
    "voice": {
        "rate": 150,        # Speaking speed
        "volume": 1.0       # Volume level
    }
}


# ============================================================================
# VOICE INTERFACE - Handles all voice I/O
# ============================================================================

class VoiceInterface:
    """Manages all voice interactions"""
    
    def __init__(self, config: dict):
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', config['voice']['rate'])
        self.tts.setProperty('volume', config['voice']['volume'])
        
        # Set voice to female if available
        voices = self.tts.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts.setProperty('voice', voice.id)
                break
    
    def speak(self, text: str, show_text: bool = True):
        """Convert text to speech"""
        if show_text:
            print(f"\n🤖 Assistant: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen(self, prompt: Optional[str] = None, timeout: int = 10) -> str:
        """Convert speech to text"""
        if prompt:
            self.speak(prompt)
        
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
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
                self.speak("Sorry, I couldn't understand that. Could you repeat?")
                return ""
            except Exception as e:
                print(f"❌ Error: {e}")
                return ""
    
    def confirm(self, question: str) -> bool:
        """Ask yes/no confirmation"""
        response = self.listen(question)
        return any(word in response.lower() for word in ["yes", "yeah", "yep", "sure", "okay", "ok", "correct", "right"])
    
    def extract_numbers(self, text: str) -> str:
        """Extract all numbers from text"""
        numbers = re.findall(r'\d+', text)
        return ''.join(numbers) if numbers else ""


# ============================================================================
# AI CONVERSATION MANAGER
# ============================================================================

class ConversationAI:
    """Handles intelligent conversation with OpenAI"""
    
    def __init__(self, api_key: str, voice: VoiceInterface):
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.history = [{
            "role": "system",
            "content": """You are a friendly Australian grocery shopping assistant.

Your job:
1. Understand what items the user wants to buy
2. Extract specific product names and quantities
3. Be conversational and ask clarifying questions
4. Help users be specific (e.g., "What type of milk? Full cream, skim, or almond?")

Always respond in a natural, friendly way. Keep responses concise for voice interaction."""
        }]
    
    def chat(self, user_message: str) -> str:
        """Get AI response"""
        self.history.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                temperature=0.7,
                max_tokens=250
            )
            
            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            return reply
            
        except Exception as e:
            print(f"❌ AI Error: {e}")
            return "I'm having trouble understanding. Could you rephrase that?"
    
    def extract_order_details(self, conversation: str) -> List[Dict]:
        """Extract items and quantities from conversation"""
        try:
            prompt = f"""Based on this conversation, extract the items and quantities the user wants to buy.

Conversation: {conversation}

Respond ONLY with a JSON array like this:
[{{"item": "milk", "quantity": 2, "details": "full cream"}}, {{"item": "bread", "quantity": 1, "details": "wholemeal"}}]

If no specific items mentioned yet, respond with: []"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            import json
            result = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            result = result.replace("```json", "").replace("```", "").strip()
            return json.loads(result)
            
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return []


# ============================================================================
# SUPERMARKET AUTOMATION
# ============================================================================

class SupermarketAutomation:
    """Handles browser automation for Coles and Woolworths"""
    
    def __init__(self, config: dict, voice: VoiceInterface):
        self.config = config
        self.voice = voice
        self.driver = None
        
        # Chrome options
        self.chrome_options = Options()
        if config['browser']['headless']:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    def create_driver(self) -> webdriver.Chrome:
        """Create browser instance"""
        driver = webdriver.Chrome(options=self.chrome_options)
        driver.implicitly_wait(self.config['browser']['timeout'])
        driver.maximize_window()
        return driver
    
    def login_coles(self) -> bool:
        """Login to Coles"""
        try:
            self.voice.speak("Logging into your Coles account...")
            self.driver = self.create_driver()
            
            self.driver.get(self.config['coles']['url'])
            time.sleep(3)
            
            # Click login button
            login_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Log in')] | //a[contains(., 'Log in')]"))
            )
            login_btn.click()
            time.sleep(2)
            
            # Enter email
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "email"))
            )
            email_field.clear()
            email_field.send_keys(self.config['coles']['username'])
            
            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.clear()
            password_field.send_keys(self.config['coles']['password'])
            
            # Submit
            submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()
            time.sleep(5)
            
            self.voice.speak("Successfully logged into Coles")
            return True
            
        except Exception as e:
            self.voice.speak("Sorry, I couldn't log into Coles. There might be a connection issue.")
            print(f"❌ Coles login error: {e}")
            return False
    
    def login_woolworths(self) -> bool:
        """Login to Woolworths"""
        try:
            self.voice.speak("Logging into your Woolworths account...")
            self.driver = self.create_driver()
            
            self.driver.get(self.config['woolworths']['url'])
            time.sleep(3)
            
            # Click login
            login_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Log in')] | //a[contains(., 'Log in')]"))
            )
            login_btn.click()
            time.sleep(2)
            
            # Enter credentials
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "loginForm-email"))
            )
            email_field.send_keys(self.config['woolworths']['username'])
            
            password_field = self.driver.find_element(By.ID, "loginForm-password")
            password_field.send_keys(self.config['woolworths']['password'])
            
            # Submit
            submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()
            time.sleep(5)
            
            self.voice.speak("Successfully logged into Woolworths")
            return True
            
        except Exception as e:
            self.voice.speak("Sorry, I couldn't log into Woolworths. There might be a connection issue.")
            print(f"❌ Woolworths login error: {e}")
            return False
    
    def search_and_add_item(self, item_name: str, quantity: int, store: str) -> bool:
        """Search for item and add to cart"""
        try:
            self.voice.speak(f"Searching for {item_name}")
            
            # Find search box
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'], input[placeholder*='Search']"))
            )
            search_box.clear()
            search_box.send_keys(item_name)
            search_box.send_keys(Keys.RETURN)
            time.sleep(4)
            
            # Click first product
            self.voice.speak(f"Found {item_name}. Adding to cart.")
            first_product = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "section[data-testid='product-tile'], article.product, div.product-tile"))
            )
            first_product.click()
            time.sleep(3)
            
            # Find and click Add button multiple times for quantity
            add_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Add') and not(contains(., 'Added'))] | //button[@aria-label='Add']"))
            )
            
            for i in range(quantity):
                try:
                    add_button.click()
                    time.sleep(1)
                    print(f"  Added {i+1}/{quantity}")
                except:
                    # Try to find increment button if item already in cart
                    try:
                        increment = self.driver.find_element(By.XPATH, "//button[@aria-label='Increase quantity'] | //button[contains(@class, 'increase')]")
                        increment.click()
                        time.sleep(1)
                    except:
                        pass
            
            self.voice.speak(f"Successfully added {quantity} {item_name} to your cart")
            
            # Go back to home
            self.driver.get(self.config[store.lower()]['url'])
            time.sleep(2)
            
            return True
            
        except Exception as e:
            self.voice.speak(f"Sorry, I couldn't add {item_name}. It might not be available.")
            print(f"❌ Add item error: {e}")
            return False
    
    def get_delivery_options(self) -> List[str]:
        """Extract delivery options from checkout page"""
        try:
            # Look for delivery/pickup options
            options = []
            
            delivery_elements = self.driver.find_elements(By.XPATH, 
                "//label[contains(., 'Delivery')] | //label[contains(., 'Pickup')] | //div[contains(@class, 'delivery-option')]//span"
            )
            
            for elem in delivery_elements:
                text = elem.text.strip()
                if text and len(text) > 3:
                    options.append(text)
            
            # Fallback options
            if not options:
                options = ["Home Delivery", "Click & Collect", "Express Delivery"]
            
            return options[:5]  # Return max 5 options
            
        except Exception as e:
            print(f"Delivery options error: {e}")
            return ["Home Delivery", "Click & Collect"]
    
    def select_delivery_option(self, option_index: int):
        """Select delivery option by index"""
        try:
            delivery_radios = self.driver.find_elements(By.XPATH, 
                "//input[@type='radio' and contains(@name, 'delivery')] | //button[contains(@class, 'delivery-option')]"
            )
            
            if 0 <= option_index < len(delivery_radios):
                delivery_radios[option_index].click()
                time.sleep(2)
                self.voice.speak("Delivery option selected")
                return True
            
        except Exception as e:
            print(f"Select delivery error: {e}")
        return False
    
    def get_payment_methods(self) -> List[str]:
        """Extract payment methods"""
        try:
            methods = []
            
            payment_elements = self.driver.find_elements(By.XPATH,
                "//label[contains(., 'Card')] | //label[contains(., 'PayPal')] | //div[contains(@class, 'payment')]//span"
            )
            
            for elem in payment_elements:
                text = elem.text.strip()
                if text and len(text) > 3:
                    methods.append(text)
            
            if not methods:
                methods = ["Credit/Debit Card", "PayPal", "Gift Card"]
            
            return methods
            
        except:
            return ["Credit/Debit Card", "PayPal"]
    
    def complete_payment(self, voice: VoiceInterface):
        """Complete payment with voice input"""
        try:
            # Navigate to checkout
            self.voice.speak("Proceeding to checkout")
            
            cart_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'cart')] | //a[contains(@href, 'trolley')]"))
            )
            cart_btn.click()
            time.sleep(3)
            
            checkout_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Checkout')] | //a[contains(., 'Checkout')]"))
            )
            checkout_btn.click()
            time.sleep(5)
            
            # === DELIVERY OPTIONS ===
            self.voice.speak("Let me check the delivery options available")
            time.sleep(2)
            
            delivery_options = self.get_delivery_options()
            
            options_text = "I can see these delivery options: " + ", ".join([f"Option {i+1}, {opt}" for i, opt in enumerate(delivery_options)])
            self.voice.speak(options_text)
            
            delivery_choice = voice.listen("Which delivery option would you like? Say the option number.")
            
            # Parse selection
            selected_delivery = 0
            for i, option in enumerate(delivery_options):
                if str(i+1) in delivery_choice or option.lower() in delivery_choice.lower():
                    selected_delivery = i
                    break
            
            self.voice.speak(f"Selecting {delivery_options[selected_delivery]}")
            self.select_delivery_option(selected_delivery)
            
            # Continue to payment
            try:
                continue_btn = self.driver.find_element(By.XPATH, 
                    "//button[contains(., 'Continue')] | //button[contains(., 'Next')] | //button[contains(., 'Proceed')]"
                )
                continue_btn.click()
                time.sleep(3)
            except:
                pass
            
            # === PAYMENT METHODS ===
            self.voice.speak("Now let me check the payment options")
            time.sleep(2)
            
            payment_methods = self.get_payment_methods()
            
            methods_text = "Available payment methods: " + ", ".join([f"Option {i+1}, {m}" for i, m in enumerate(payment_methods)])
            self.voice.speak(methods_text)
            
            payment_choice = voice.listen("Which payment method would you like to use? Say the option number.")
            
            # Parse payment selection
            selected_payment = 0
            for i, method in enumerate(payment_methods):
                if str(i+1) in payment_choice or method.lower() in payment_choice.lower():
                    selected_payment = i
                    break
            
            self.voice.speak(f"Selected {payment_methods[selected_payment]}")
            
            # Click payment option
            try:
                payment_radios = self.driver.find_elements(By.XPATH,
                    "//input[@type='radio' and contains(@name, 'payment')]"
                )
                if selected_payment < len(payment_radios):
                    payment_radios[selected_payment].click()
                    time.sleep(2)
            except:
                pass
            
            # === CARD DETAILS ===
            if voice.confirm("Do you need to enter new card details?"):
                
                # Card number
                card_num = voice.listen("Please say your card number, digit by digit")
                card_number = voice.extract_numbers(card_num)
                
                if card_number:
                    try:
                        card_field = self.driver.find_element(By.XPATH,
                            "//input[@name='cardNumber' or @placeholder*='card number' or @id='cardNumber' or contains(@id, 'card-number')]"
                        )
                        card_field.clear()
                        card_field.send_keys(card_number)
                        self.voice.speak("Card number entered")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Card number error: {e}")
                
                # Card name
                card_name = voice.listen("Please say the name on the card")
                if card_name:
                    try:
                        name_field = self.driver.find_element(By.XPATH,
                            "//input[@name='cardName' or @placeholder*='name' or @id='cardName' or contains(@id, 'card-name')]"
                        )
                        name_field.clear()
                        name_field.send_keys(card_name)
                        self.voice.speak("Card name entered")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Card name error: {e}")
                
                # Expiry
                expiry = voice.listen("Please say the expiry date, month and year")
                exp_numbers = voice.extract_numbers(expiry)
                
                if len(exp_numbers) >= 4:
                    expiry_formatted = f"{exp_numbers[:2]}/{exp_numbers[2:]}"
                    try:
                        expiry_field = self.driver.find_element(By.XPATH,
                            "//input[@name='expiryDate' or @placeholder*='expiry' or @id='expiryDate' or contains(@id, 'expiry')]"
                        )
                        expiry_field.clear()
                        expiry_field.send_keys(expiry_formatted)
                        self.voice.speak("Expiry date entered")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Expiry error: {e}")
            
            # === CVV (Always required) ===
            cvv = voice.listen("Please say your CVV code, the three digit security code")
            cvv_code = voice.extract_numbers(cvv)
            
            if cvv_code:
                try:
                    cvv_field = self.driver.find_element(By.XPATH,
                        "//input[@name='cvv' or @name='securityCode' or @placeholder*='CVV' or @placeholder*='Security' or @id='cvv' or contains(@id, 'cvv') or contains(@id, 'security-code')]"
                    )
                    cvv_field.clear()
                    cvv_field.send_keys(cvv_code)
                    self.voice.speak("CVV entered successfully")
                    time.sleep(1)
                except Exception as e:
                    self.voice.speak("I couldn't find the CVV field. Please enter it manually if needed.")
                    print(f"CVV error: {e}")
            
            # === PLACE ORDER ===
            if voice.confirm("Everything is ready. Should I place the order now?"):
                try:
                    place_order_btn = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH,
                            "//button[contains(., 'Place order')] | //button[contains(., 'Pay now')] | //button[contains(., 'Complete')] | //button[contains(., 'Confirm')]"
                        ))
                    )
                    place_order_btn.click()
                    self.voice.speak("Order submitted! Checking for verification...")
                    time.sleep(5)
                    
                    # === OTP VERIFICATION ===
                    try:
                        otp_field = self.driver.find_element(By.XPATH,
                            "//input[@name='otp' or @placeholder*='OTP' or @placeholder*='code' or @placeholder*='verification' or contains(@id, 'otp')]"
                        )
                        
                        otp = voice.listen("I see a verification code is required. Please say your OTP code")
                        otp_code = voice.extract_numbers(otp)
                        
                        if otp_code:
                            otp_field.clear()
                            otp_field.send_keys(otp_code)
                            self.voice.speak("OTP entered")
                            time.sleep(1)
                            
                            # Submit OTP
                            otp_submit = self.driver.find_element(By.XPATH,
                                "//button[@type='submit'] | //button[contains(., 'Verify')] | //button[contains(., 'Submit')]"
                            )
                            otp_submit.click()
                            time.sleep(3)
                        
                        self.voice.speak("Order completed successfully! You should receive a confirmation email shortly.")
                        
                    except:
                        # No OTP required
                        self.voice.speak("Order completed successfully! You should receive a confirmation email shortly.")
                    
                    return True
                    
                except Exception as e:
                    self.voice.speak("I couldn't complete the order. Please check the screen for any issues.")
                    print(f"Place order error: {e}")
                    return False
            else:
                self.voice.speak("Order cancelled. Your cart is still saved if you want to checkout later.")
                return False
                
        except Exception as e:
            self.voice.speak("There was an issue during checkout. Please check the browser.")
            print(f"Checkout error: {e}")
            return False
    
    def cleanup(self):
        """Close browser"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


# ============================================================================
# MAIN VOICE SHOPPING ASSISTANT
# ============================================================================

class VoiceShoppingAssistant:
    """Main assistant orchestrating the entire shopping experience"""
    
    def __init__(self, config: dict):
        print("\n" + "="*70)
        print("  VOICE SHOPPING ASSISTANT - FINAL VERSION")
        print("  Fully Voice-Controlled Grocery Shopping")
        print("="*70 + "\n")
        
        if config['openai_api_key'] == "PUT-YOUR-OPENAI-API-KEY-HERE":
            raise ValueError("\n❌ Please add your OpenAI API key in the CONFIG section!\n")
        
        # Initialize components
        print("🔧 Initializing voice systems...")
        self.voice = VoiceInterface(config)
        print("✓ Voice ready")
        
        print("🔧 Connecting to AI...")
        self.ai = ConversationAI(config['openai_api_key'], self.voice)
        print("✓ AI ready")
        
        print("🔧 Setting up browser automation...")
        self.automation = SupermarketAutomation(config, self.voice)
        print("✓ Browser ready")
        
        self.config = config
        self.cart = []
        
        print("\n✅ All systems ready! Starting assistant...\n")
        time.sleep(1)
    
    def run(self):
        """Main conversation flow"""
        
        try:
            # === STEP 1: GREET USER ===
            self.voice.speak("Hello! Welcome to your voice shopping assistant.")
            self.voice.speak("I can help you order groceries from Coles and Woolworths completely hands-free.")
            time.sleep(1)
            
            # === STEP 2: ASK FOR ORDER & UNDERSTAND NEEDS ===
            self.voice.speak("What would you like to buy today?")
            
            conversation_log = []
            order_confirmed = False
            
            while not order_confirmed:
                user_input = self.voice.listen()
                
                if not user_input:
                    continue
                
                conversation_log.append(user_input)
                
                # Check if user wants to exit
                if any(word in user_input.lower() for word in ["exit", "quit", "cancel", "nevermind", "stop"]):
                    self.voice.speak("No problem! Have a great day!")
                    return
                
                # Get AI response to understand order better
                ai_response = self.ai.chat(user_input)
                self.voice.speak(ai_response)
                
                # Check if we have enough information
                if any(phrase in ai_response.lower() for phrase in ["got it", "understand", "perfect", "great"]):
                    # Try to extract order details
                    full_conversation = " ".join(conversation_log)
                    order_details = self.ai.extract_order_details(full_conversation)
                    
                    if order_details:
                        # Confirm order
                        self.voice.speak("Let me confirm your order:")
                        for item in order_details:
                            details_text = f"{item['quantity']} {item['item']}"
                            if item.get('details'):
                                details_text += f", {item['details']}"
                            self.voice.speak(details_text)
                        
                        if self.voice.confirm("Is this correct?"):
                            self.cart = order_details
                            order_confirmed = True
                        else:
                            self.voice.speak("No problem. What would you like to change?")
            
            # === STEP 3: ASK FOR STORE ===
            store_choice = self.voice.listen("Great! Would you like to order from Coles or Woolworths?")
            
            if "coles" in store_choice.lower():
                store = "coles"
                store_name = "Coles"
            elif "woolworths" in store_choice.lower() or "woolworth" in store_choice.lower():
                store = "woolworths"
                store_name = "Woolworths"
            else:
                # Ask again
                clarify = self.voice.listen("I didn't catch that. Please say Coles or Woolworths")
                store = "coles" if "coles" in clarify.lower() else "woolworths"
                store_name = "Coles" if store == "coles" else "Woolworths"
            
            self.voice.speak(f"Perfect! I'll order from {store_name}")
            
            # === STEP 4: LOGIN AND ADD ITEMS TO CART ===
            if store == "coles":
                login_success = self.automation.login_coles()
            else:
                login_success = self.automation.login_woolworths()
            
            if not login_success:
                self.voice.speak("Sorry, I couldn't log in to your account. Please check your credentials and try again.")
                return
            
            # Add each item to cart
            self.voice.speak(f"Now I'll add your items to the {store_name} cart")
            
            for item in self.cart:
                item_description = item['item']
                if item.get('details'):
                    item_description += f" {item['details']}"
                
                success = self.automation.search_and_add_item(
                    item_description,
                    item['quantity'],
                    store
                )
                
                if not success:
                    if not self.voice.confirm(f"Should I continue with the other items?"):
                        self.voice.speak("Understood. Stopping here.")
                        return
                
                time.sleep(2)  # Brief pause between items
            
            self.voice.speak("All items have been added to your cart!")
            
            # === STEP 5 & 6: DELIVERY OPTIONS & PAYMENT ===
            if self.voice.confirm("Would you like to proceed to checkout now?"):
                self.automation.complete_payment(self.voice)
            else:
                self.voice.speak("No problem! Your items are saved in your cart. You can checkout later from the website.")
            
            # === COMPLETION ===
            self.voice.speak("Thank you for using the voice shopping assistant! Have a wonderful day!")
            
        except KeyboardInterrupt:
            self.voice.speak("Stopping assistant. Goodbye!")
        
        except Exception as e:
            self.voice.speak("Sorry, something went wrong. Please try again.")
            print(f"\n❌ Error: {e}")
        
        finally:
            self.automation.cleanup()


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("  VOICE SHOPPING ASSISTANT")
    print("  Checking system requirements...")
    print("="*70)
    
    # Check OpenAI API key
    if CONFIG['openai_api_key'] == "PUT-YOUR-OPENAI-API-KEY-HERE":
        print("\n❌ ERROR: OpenAI API key not configured!")
        print("\n📝 To fix this:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Copy the key (starts with 'sk-')")
        print("4. Edit this file and paste it on line 26")
        print("5. Replace 'PUT-YOUR-OPENAI-API-KEY-HERE' with your key\n")
        input("Press Enter to exit...")
        exit(1)
    
    # Check ChromeDriver
    print("\n🔍 Checking ChromeDriver...")
    try:
        test_driver = webdriver.Chrome()
        test_driver.quit()
        print("✓ ChromeDriver found and working\n")
    except Exception as e:
        print(f"\n❌ ChromeDriver not found or not working!")
        print(f"Error: {e}\n")
        print("📥 To install ChromeDriver:")
        print("   Method 1: pip install webdriver-manager")
        print("   Method 2: Download from https://chromedriver.chromium.org/")
        print("   Make sure Chrome browser is installed\n")
        input("Press Enter to exit...")
        exit(1)
    
    # Check microphone
    print("🎤 Checking microphone...")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("✓ Microphone detected\n")
    except Exception as e:
        print(f"⚠️  Microphone issue: {e}")
        print("Make sure a microphone is connected and permissions are granted\n")
    
    # Start assistant
    print("🚀 Starting Voice Shopping Assistant...\n")
    time.sleep(1)
    
    try:
        assistant = VoiceShoppingAssistant(CONFIG)
        assistant.run()
    
    except KeyboardInterrupt:
        print("\n\n👋 Program stopped by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        print("\nIf this keeps happening, please check:")
        print("- Your internet connection")
        print("- OpenAI API key is valid")
        print("- Microphone is working")
        print("- Chrome and ChromeDriver are compatible versions\n")
        input("Press Enter to exit...")