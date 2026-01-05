import speech_recognition as sr
import pyttsx3
import os
import time
import re
import json
from typing import Optional, List, Dict

from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ============================================================
# CONFIG (API KEY ENV VAR)
# ============================================================

from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),

    "coles": {
        "username": os.getenv("COLES_USERNAME", ""),
        "password": os.getenv("COLES_PASSWORD", ""),
        "enabled": os.getenv("COLES_ENABLED", "true").lower() in ("1", "true", "yes"),
        "url": os.getenv("COLES_URL", "https://www.coles.com.au/")
    },

    "woolworths": {
        "username": os.getenv("WOOLWORTHS_USERNAME", ""),
        "password": os.getenv("WOOLWORTHS_PASSWORD", ""),
        "enabled": os.getenv("WOOLWORTHS_ENABLED", "true").lower() in ("1", "true", "yes"),
        "url": os.getenv("WOOLWORTHS_URL", "https://www.woolworths.com.au/")
    },

    "browser": {
        "headless": os.getenv("BROWSER_HEADLESS", "false").lower() in ("1", "true", "yes"),
        "timeout": int(os.getenv("BROWSER_TIMEOUT", "20"))
    },

    "voice": {
        "rate": int(os.getenv("VOICE_RATE", "150")),        # Speaking speed
        "volume": float(os.getenv("VOICE_VOLUME", "1.0"))   # Volume level
    }
}

if not CONFIG["openai_api_key"]:
    raise RuntimeError("OPENAI_API_KEY not set")

# ============================================================
# VOICE INTERFACE (BUG FIXED)
# ============================================================

class VoiceInterface:
    def __init__(self, config):
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300
        self.rate = config["voice"]["rate"]
        self.volume = config["voice"]["volume"]

    def _create_tts(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        return engine

    def speak(self, text: str):
        print(f"\n🤖 Assistant: {text}")
        engine = self._create_tts()      # 🔑 NEW ENGINE EACH TIME
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine                       # 🔑 FORCE RELEASE
        time.sleep(0.5)                  # 🔑 MIC RESET TIME

    def listen(self, prompt: Optional[str] = None, retries: int = 3) -> str:
        if prompt:
            self.speak(prompt)

        for _ in range(retries):
            try:
                with sr.Microphone() as source:
                    print("🎤 Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=20
                    )

                text = self.recognizer.recognize_google(audio)
                print(f"👤 You: {text}")
                return text.lower()

            except sr.WaitTimeoutError:
                self.speak("I didn’t hear anything. Please repeat.")
            except sr.UnknownValueError:
                self.speak("Sorry, I didn’t catch that. Please say it again.")
            except Exception as e:
                print("Voice error:", e)
                self.speak("Microphone issue. Try again.")

        return ""

    def confirm(self, question: str) -> bool:
        for _ in range(3):
            ans = self.listen(question)
            if ans:
                return any(w in ans for w in ["yes", "yeah", "ok", "sure", "correct"])
        return False

    def extract_number(self, text: str) -> int:
        nums = re.findall(r"\d+", text)
        return int(nums[0]) if nums else 1


# ============================================================
# OPENAI BRAIN
# ============================================================

class ConversationAI:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def extract_item(self, text: str) -> Dict:
        prompt = f"""
Extract grocery intent.

Return ONLY JSON:
{{"item":"milk","details":null}}

User: {text}
"""
        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(res.choices[0].message.content)

# ============================================================
# SELENIUM AUTOMATION
# ============================================================

class SupermarketAutomation:
    def __init__(self, config, voice):
        self.voice = voice
        self.config = config
        self.driver = None

    def create_driver(self):
        service = Service(ChromeDriverManager().install())
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(self.config["browser"]["timeout"])
        driver.maximize_window()
        return driver

    def login(self, store: str):
        creds = self.config[store]
        self.voice.speak(f"Logging into {store}")

        self.driver = self.create_driver()
        self.driver.get(creds["url"])
        time.sleep(5)

    # Click login button
        login_btn = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//a[contains(., 'Log in')] | //button[contains(., 'Log in')]")
         )
        )
        login_btn.click()
        time.sleep(3)

    # USERNAME FIELD
        email_input = WebDriverWait(self.driver, 15).until(
            EC.visibility_of_element_located(
                (By.XPATH, "//input[@type='email' or contains(@id,'email')]")
            )
        )
        self.driver.execute_script("arguments[0].scrollIntoView(true);", email_input)
        email_input.click()
        email_input.clear()
        email_input.send_keys(creds["username"])

    # PASSWORD FIELD
        password_input = WebDriverWait(self.driver, 15).until(
            EC.visibility_of_element_located(
                (By.XPATH, "//input[@type='password' or contains(@id,'password')]")
            )
        )
        password_input.click()
        password_input.clear()
        password_input.send_keys(creds["password"])

    # SUBMIT
        submit_btn = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[@type='submit' or contains(., 'Log in')]")
            )
        )
        submit_btn.click()

        self.voice.speak("Login successful")
        time.sleep(6)


    def add_item(self, query: str, qty: int):
        self.voice.speak(f"Adding {qty} {query}")
        box = self.driver.find_element(By.CSS_SELECTOR, "input[type='search']")
        box.clear()
        box.send_keys(query)
        box.send_keys(Keys.RETURN)
        time.sleep(5)

        self.driver.find_element(By.XPATH, "//button[contains(.,'Add')]").click()
        for _ in range(qty - 1):
            time.sleep(1)
            self.driver.find_element(
                By.XPATH, "//button[contains(@aria-label,'Increase')]"
            ).click()

# ============================================================
# MAIN ASSISTANT
# ============================================================

class VoiceShoppingAssistant:
    def __init__(self):
        self.voice = VoiceInterface(CONFIG)
        self.ai = ConversationAI(CONFIG["openai_api_key"])
        self.auto = SupermarketAutomation(CONFIG, self.voice)
        self.cart = []

    def run(self):
        self.voice.speak("What would you like to buy today?")

        while True:
            user = ""
            while not user:
                user = self.voice.listen()

            item = self.ai.extract_item(user)

            qty_text = ""
            while not qty_text:
                qty_text = self.voice.listen("How many do you want?")
            item["quantity"] = self.voice.extract_number(qty_text)

            self.cart.append(item)

            if not self.voice.confirm("Do you want to add another item?"):
                break

        store = "coles" if self.voice.confirm(
            "Should I order from Coles? Say yes for Coles, no for Woolworths."
        ) else "woolworths"

        self.auto.login(store)

        for item in self.cart:
            query = f"{item.get('details','')} {item['item']}"
            self.auto.add_item(query, item["quantity"])

        self.voice.speak("All items added to cart.")

        if self.voice.confirm("Should I proceed to checkout now?"):
            self.voice.speak("Checkout ready. Please complete payment on screen.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    VoiceShoppingAssistant().run()
