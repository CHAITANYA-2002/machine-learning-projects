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
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from pathlib import Path
from datetime import datetime
# Load .env values into the environment if present
from dotenv import load_dotenv
load_dotenv()
import time
import re

# ============================================================================
# CONFIGURATION - ADD YOUR OPENAI API KEY
# ============================================================================

CONFIG = {
    # IMPORTANT: Do NOT hard-code secrets in source. Use environment variables
    # or a secure secrets manager in production. The lines below pull sensitive
    # values from the environment and provide safe placeholders for local testing.
    "openai_api_key": os.getenv("OPENAI_API_KEY", "PUT-YOUR-OPENAI-API-KEY-HERE"),

    "coles": {
        "username": os.getenv("COLES_USERNAME", "your-coles-email@example.com"),
        "password": os.getenv("COLES_PASSWORD", "your-coles-password"),
        "enabled": False,
        "url": "https://www.coles.com.au/"
    },

    "woolworths": {
        "username": os.getenv("WOOLWORTHS_USERNAME", "your-woolworths-email@example.com"),
        "password": os.getenv("WOOLWORTHS_PASSWORD", "your-woolworths-password"),
        "enabled": False,
        "url": "https://www.woolworths.com.au/"
    },

    "browser": {
        "headless": os.getenv('BROWSER_HEADLESS', '0').lower() in ('1', 'true', 'yes'),  # Set via .env or env vars
        "timeout": int(os.getenv('BROWSER_TIMEOUT', '20'))
    },

    "voice": {
        "rate": int(os.getenv('VOICE_RATE', '150')),        # Speaking speed
        "volume": float(os.getenv('VOICE_VOLUME', '1.0'))       # Volume level
    }

}

# ============================================================================
# VOICE INTERFACE - Handles all voice I/O
# ============================================================================

class VoiceInterface:
    """Manages all voice interactions"""
    
    def __init__(self, config: dict):
        self.recognizer = sr.Recognizer()
        self.tts = None
        self.driver_used = None

        # Try multiple backend drivers to improve cross-platform reliability
        driver_candidates = [None, 'sapi5']  # None = default driver, 'sapi5' is Windows SAPI
        for drv in driver_candidates:
            try:
                engine = pyttsx3.init(driverName=drv) if drv else pyttsx3.init()
                engine.setProperty('rate', config['voice']['rate'])
                engine.setProperty('volume', config['voice']['volume'])

                # Prefer a female voice if available
                try:
                    voices = engine.getProperty('voices')
                    for v in voices:
                        if 'female' in v.name.lower() or 'zira' in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
                except Exception:
                    pass

                self.tts = engine
                self.driver_used = drv if drv else 'default'
                print(f"Initialized TTS engine using driver: {self.driver_used}")
                break
            except Exception as e:
                print(f"Unable to initialize TTS driver '{drv}': {e}")
                continue

        if not self.tts:
            print("WARNING: No TTS engine initialized. Speech will be silent until a TTS driver is available.")

    def tts_diagnostics(self) -> None:
        """Print diagnostics and attempt to play a short test phrase."""
        print("=== TTS diagnostics ===")
        print(f"Driver used: {self.driver_used}")
        try:
            if self.tts:
                try:
                    voices = self.tts.getProperty('voices')
                    print("Available voices:")
                    for v in voices:
                        print(f" - {v.name} (id={v.id})")
                except Exception as e:
                    print(f"Could not list voices: {e}")

                try:
                    print("Rate:", self.tts.getProperty('rate'))
                    print("Volume:", self.tts.getProperty('volume'))
                except Exception:
                    pass
            else:
                print("No pyttsx3 engine is currently initialized.")
        except Exception as e:
            print(f"TTS diagnostics read error: {e}")

        # Try speaking a short phrase using a short-lived engine
        try:
            test_phrase = "This is a TTS diagnostic test. Please listen for a short voice message."
            print("Attempting to play a test phrase...")
            try:
                engine = pyttsx3.init(driverName=self.driver_used if self.driver_used and self.driver_used != 'default' else None)
            except Exception:
                engine = pyttsx3.init()

            try:
                engine.say(test_phrase)
                engine.runAndWait()
                print("If you heard the test phrase, TTS is functioning.")
            except Exception as e:
                print(f"Failed to play test phrase: {e}")
            finally:
                try:
                    engine.stop()
                    del engine
                except Exception:
                    pass
        except Exception as e:
            print(f"TTS diagnostics failure: {e}")
    
    def speak(self, text: str, show_text: bool = True):
        """Convert text to speech.

        Uses a short-lived pyttsx3 engine to avoid audio device/resource locking
        issues that can cause speech to only be played once on some systems.
        """
        if show_text:
            print(f"\n🤖 Assistant: {text}")

        try:
            # Stop any currently-playing persistent engine (defensive)
            try:
                self.tts.stop()
            except Exception:
                pass

            # Create a fresh engine per utterance to avoid driver/resource state bugs
            engine = pyttsx3.init()
            # copy main engine's voice properties if available
            try:
                engine.setProperty('rate', self.tts.getProperty('rate'))
                engine.setProperty('volume', self.tts.getProperty('volume'))
                engine.setProperty('voice', self.tts.getProperty('voice'))
            except Exception:
                pass

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

            # Small pause to ensure microphone/device stability before next listen
            time.sleep(0.25)

        except Exception as e:
            print(f"TTS error: {e}")
            try:
                print("Running TTS diagnostics to identify issues...")
                self.tts_diagnostics()
                print("If you still cannot hear audio, check system volume/output device and that a TTS backend (e.g., SAPI5 on Windows) is available.")
            except Exception:
                pass

    def listen(self, prompt: Optional[str] = None, timeout: int = 10) -> str:
        """Convert speech to text. Adds short delays to avoid picking up TTS audio."""
        if prompt:
            self.speak(prompt)

        # Ensure any TTS audio has finished and stop persistent engine as a fallback
        try:
            self.tts.stop()
        except Exception:
            pass
        time.sleep(0.25)

        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            # Give microphone a moment to settle after TTS playback
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

        # Logs directory for debugging artifacts (screenshots, page source)
        self.logs_dir = Path("voiceshopping/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Allow enabling/disabling stores via environment variables for safety
        self.config['coles']['enabled'] = os.getenv('COLES_ENABLED', str(self.config['coles'].get('enabled', False))).lower() in ('1', 'true', 'yes')
        self.config['woolworths']['enabled'] = os.getenv('WOOLWORTHS_ENABLED', str(self.config['woolworths'].get('enabled', False))).lower() in ('1', 'true', 'yes')

        # Chrome options
        self.chrome_options = Options()
        if config['browser']['headless']:
            # Keep headless off by default for debugging login flows
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    


    def create_driver(self) -> webdriver.Chrome:
        """Create a Chrome WebDriver using webdriver-manager.

        Returns a working Chrome WebDriver instance or raises an exception.
        """
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.chrome_options)
        driver.implicitly_wait(self.config['browser']['timeout'])
        try:
            driver.maximize_window()
        except Exception:
            # Not all environments support maximize (headless servers)
            pass
        return driver

    # ---------- Debug helpers ----------
    def _take_debug_artifacts(self, label: str) -> None:
        """Save a screenshot and page source to the logs directory for debugging."""
        try:
            if not self.driver:
                return
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot = self.logs_dir / f"{ts}_{label}.png"
            page_html = self.logs_dir / f"{ts}_{label}.html"
            self.driver.save_screenshot(str(screenshot))
            page_html.write_text(self.driver.page_source, encoding='utf-8')
            print(f"Saved debug artifacts: {screenshot}, {page_html}")
        except Exception as e:
            print(f"Failed to save debug artifacts: {e}")

    def _close_cookie_banner(self) -> None:
        """Try to close common cookie/privacy banners that block login elements."""
        selectors = [
            "//button[contains(., 'Accept') or contains(., 'Agree') or contains(., 'OK') or contains(., 'Allow')]",
            "//button[contains(., 'Dismiss') or contains(., 'Close')]",
            "//a[contains(., 'Accept') or contains(., 'Continue')]",
        ]
        for sel in selectors:
            try:
                btn = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((By.XPATH, sel)))
                btn.click()
                print("Closed cookie/privacy banner")
                time.sleep(1)
            except Exception:
                pass

    def _find_input(self, possible_xpaths: list, timeout: int = 6):
        """Try multiple xpaths to find an input element; return the first found."""
        for xp in possible_xpaths:
            try:
                elm = WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, xp)))
                return elm
            except Exception:
                continue
        return None

    def _is_logged_in(self, store: str) -> bool:
        """Heuristics to determine whether login succeeded by looking for account indicators."""
        checks = [
            "//a[contains(., 'Sign out') or contains(., 'Log out') or contains(., 'Sign Out')]",
            "//a[contains(., 'My Account') or contains(., 'Account')]",
            "//span[contains(@class,'user') or contains(@class,'account')]",
            "//div[contains(@class,'account')]",
        ]
        for xp in checks:
            try:
                if self.driver.find_elements(By.XPATH, xp):
                    return True
            except Exception:
                continue
        return False

    
    def login_coles(self) -> bool:
        """Login to Coles with robust selectors, cookie handling, and diagnostics."""
        if not self.config['coles'].get('enabled', False):
            self.voice.speak("Coles automation is disabled. Enable it via COLES_ENABLED environment variable or in CONFIG.")
            return False

        username = self.config['coles'].get('username')
        password = self.config['coles'].get('password')
        if not username or 'your-coles' in username or not password or 'your-coles' in password:
            self.voice.speak("Coles credentials are not configured. Please set COLES_USERNAME and COLES_PASSWORD environment variables.")
            return False

        try:
            self.voice.speak("Logging into your Coles account...")
            self.driver = self.create_driver()
            
            self.driver.get(self.config['coles']['url'])
            time.sleep(2)

            # Attempt to close cookie/privacy banners that might block UI and try several login button selectors
            self._close_cookie_banner()

            login_selectors = [
                "//button[contains(., 'Log in')]",
                "//a[contains(., 'Log in')]",
                "//button[contains(., 'Sign in')]",
                "//a[contains(., 'Sign in')]",
                "//button[contains(@aria-label,'log in') or contains(@aria-label,'sign in')]",
            ]
            login_btn = None
            for xp in login_selectors:
                try:
                    login_btn = WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable((By.XPATH, xp)))
                    break
                except Exception:
                    continue

            if login_btn:
                login_btn.click()
                time.sleep(1)

            # Find email input with fallbacks
            # Find email input with fallbacks
            email_xpaths = [
                "//input[@id='email']",
                "//input[@name='email']",
                "//input[@type='email']",
                "//input[contains(@placeholder, 'Email') or contains(@placeholder, 'email')]",
            ]
            email_field = self._find_input(email_xpaths)
            if email_field:
                email_field.clear()
                email_field.send_keys(self.config['coles']['username'])
            else:
                raise RuntimeError('Email input not found on Coles login page')
            # Enter password with fallbacks
            password_xpaths = [
                "//input[@id='password']",
                "//input[@name='password']",
                "//input[@type='password']",
                "//input[contains(@placeholder, 'Password') or contains(@placeholder, 'password')]",
            ]
            password_field = self._find_input(password_xpaths)
            if password_field:
                password_field.clear()
                password_field.send_keys(self.config['coles']['password'])
            else:
                raise RuntimeError('Password input not found on Coles login page')

            # Try to submit using button or ENTER key
            try:
                submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                submit_btn.click()
            except Exception:
                password_field.send_keys(Keys.RETURN)

            # Wait a bit and check logged-in status
            time.sleep(4)
            if self._is_logged_in('coles'):
                self.voice.speak("Successfully logged into Coles")
                return True
            else:
                self._take_debug_artifacts('coles_login_failed')
                self.voice.speak("Login to Coles failed - I saved a screenshot for debugging.")
                return False

        except Exception as e:
            print(f"❌ Coles login error: {e}")
            try:
                self._take_debug_artifacts('coles_exception')
            except Exception:
                pass
            self.voice.speak("Sorry, I couldn't log into Coles. Check credentials and try again.")
            return False
    
    def login_woolworths(self) -> bool:
        """Login to Woolworths with robust selectors and diagnostics."""
        if not self.config['woolworths'].get('enabled', False):
            self.voice.speak("Woolworths automation is disabled. Enable it via WOOLWORTHS_ENABLED environment variable or in CONFIG.")
            return False

        username = self.config['woolworths'].get('username')
        password = self.config['woolworths'].get('password')
        if not username or 'your-woolworths' in username or not password or 'your-woolworths' in password:
            self.voice.speak("Woolworths credentials are not configured. Please set WOOLWORTHS_USERNAME and WOOLWORTHS_PASSWORD environment variables.")
            return False

        try:
            self.voice.speak("Logging into your Woolworths account...")
            self.driver = self.create_driver()

            self.driver.get(self.config['woolworths']['url'])
            time.sleep(2)

            self._close_cookie_banner()

            # Try a set of possible login selectors and click
            login_selectors = [
                "//button[contains(., 'Log in')]",
                "//a[contains(., 'Log in')]",
                "//button[contains(., 'Sign in')]",
                "//a[contains(., 'Sign in')]",
            ]
            login_btn = None
            for xp in login_selectors:
                try:
                    login_btn = WebDriverWait(self.driver, 6).until(EC.element_to_be_clickable((By.XPATH, xp)))
                    break
                except Exception:
                    continue

            if login_btn:
                login_btn.click()
                time.sleep(1)

            # Find email and password inputs with fallbacks
            email_xpaths = [
                "//input[@id='loginForm-email']",
                "//input[@name='email']",
                "//input[@type='email']",
                "//input[contains(@placeholder, 'Email') or contains(@placeholder, 'email')]",
            ]
            email_field = self._find_input(email_xpaths)
            if email_field:
                email_field.clear()
                email_field.send_keys(username)
            else:
                raise RuntimeError('Email input not found on Woolworths login page')

            password_xpaths = [
                "//input[@id='loginForm-password']",
                "//input[@name='password']",
                "//input[@type='password']",
                "//input[contains(@placeholder, 'Password') or contains(@placeholder, 'password')]",
            ]
            password_field = self._find_input(password_xpaths)
            if password_field:
                password_field.clear()
                password_field.send_keys(password)
            else:
                raise RuntimeError('Password input not found on Woolworths login page')

            # Submit
            try:
                submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                submit_btn.click()
            except Exception:
                password_field.send_keys(Keys.RETURN)

            time.sleep(4)
            if self._is_logged_in('woolworths'):
                self.voice.speak("Successfully logged into Woolworths")
                return True
            else:
                self._take_debug_artifacts('woolworths_login_failed')
                self.voice.speak("Login to Woolworths failed - screenshot saved for debugging.")
                return False

        except Exception as e:
            print(f"❌ Woolworths login error: {e}")
            try:
                self._take_debug_artifacts('woolworths_exception')
            except Exception:
                pass
            self.voice.speak("Sorry, I couldn't log into Woolworths. Check credentials and try again.")
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
    
    # Check ChromeDriver - try Selenium Manager first, then fall back to webdriver-manager
    print("\n🔍 Checking ChromeDriver...")

    # Helper to try creating a driver using webdriver-manager
    def _try_with_webdriver_manager():
        try:
            opts = Options()
            # Headless helps during automated checks, but some sites require non-headless for JS
            opts.add_argument('--headless=new')
            opts.add_argument('--no-sandbox')
            opts.add_argument('--disable-dev-shm-usage')
            service = Service(ChromeDriverManager().install())
            d = webdriver.Chrome(service=service, options=opts)
            d.quit()
            return True, None
        except Exception as e:
            return False, e

    # First attempt: let Selenium Manager create a driver (webdriver.Chrome())
    try:
        test_driver = webdriver.Chrome()
        test_driver.quit()
        print("✓ ChromeDriver found and working (Selenium Manager)\n")
    except Exception as selenium_err:
        print(f"Selenium Manager failed: {selenium_err}")
        print("Attempting to obtain ChromeDriver using webdriver-manager...")

        ok, err = _try_with_webdriver_manager()
        if ok:
            print("✓ ChromeDriver installed and working via webdriver-manager\n")
        else:
            print(f"\n❌ ChromeDriver not found or not working!\nReason: {err}\n")
            print("📥 To install ChromeDriver manually:")
            print("   - Method 1: pip install webdriver-manager (recommended)")
            print("   - Method 2: Download matching ChromeDriver from https://chromedriver.chromium.org/")
            print("   - Ensure Google Chrome is installed and the driver version matches your browser")
            print("If you installed 'webdriver-manager' but still see errors, try running this script as an admin or check firewall/anti-virus settings.\n")
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

    # Validate store configuration if any store enabled
    def _validate_store_config():
        issues = []
        for store in ('coles', 'woolworths'):
            enabled = CONFIG.get(store, {}).get('enabled', False)
            if enabled:
                user = CONFIG.get(store, {}).get('username')
                pwd = CONFIG.get(store, {}).get('password')
                if not user or 'your-' in str(user) or not pwd or 'your-' in str(pwd):
                    issues.append((store, 'credentials_missing'))
        return issues

    store_issues = _validate_store_config()
    if store_issues:
        print("\n⚠️  Store configuration issues detected:")
        for s, issue in store_issues:
            print(f" - {s}: missing credentials (set via environment variables {s.upper()}_USERNAME/{s.upper()}_PASSWORD or in CONFIG)")
        print("\nYou can either set the environment variables, edit the CONFIG to add credentials, or disable the store by setting {STORE}_ENABLED=0\n")
        input("Press Enter to continue without attempting store logins... ")

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