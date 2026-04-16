import streamlit as st
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import speech_recognition as sr
from streamlit_mic_recorder import speech_to_text

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sarkari Yojana Sahayak | Government Scheme Finder",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE — initialise all keys once
# ══════════════════════════════════════════════════════════════════
defaults = {
    "history": [],
    "query":   "",
    "lang":    "hi",        # "hi" = Hindi UI  |  "en" = English UI
    "feedback": {},         # stores thumbs feedback per query
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

LANG = st.session_state.lang  # shortcut used throughout file

# ══════════════════════════════════════════════════════════════════
# UI STRING DICTIONARY  (Hindi / English)
# ══════════════════════════════════════════════════════════════════
UI = {
    "page_title":       {"hi": "सरकारी योजना सहायक", "en": "Government Scheme Finder"},
    "eyebrow":          {"hi": "🇮🇳 भारत सरकार · Government of India",
                         "en": "🇮🇳 Government of India · Bharat Sarkar"},
    "hero_title_1":     {"hi": "सरकारी", "en": "Government"},
    "hero_title_2":     {"hi": "योजना", "en": "Scheme"},
    "hero_title_3":     {"hi": "सहायक", "en": "Finder"},
    "hero_sub":         {"hi": "ग्रामीण नागरिकों के लिए — हिंदी, हिंग्लिश या English में अपनी ज़रूरत बताएं और सही सरकारी योजना खोजें",
                         "en": "For rural citizens — tell us your need in Hindi, Hinglish or English and find the right government scheme"},
    "stat_train":       {"hi": "प्रशिक्षण वाक्य", "en": "Training Sentences"},
    "stat_cat":         {"hi": "योजना श्रेणियाँ",  "en": "Scheme Categories"},
    "stat_schemes":     {"hi": "सरकारी योजनाएं",   "en": "Govt Schemes"},
    "stat_lang":        {"hi": "भाषाएं",            "en": "Languages"},
    "quick_select":     {"hi": "⚡ श्रेणी चुनें — Quick Select", "en": "⚡ Choose Category — Quick Select"},
    "cat_edu":          {"hi": "शिक्षा",   "en": "Education"},
    "cat_agri":         {"hi": "कृषि",     "en": "Agriculture"},
    "cat_emp":          {"hi": "रोजगार",   "en": "Employment"},
    "cat_women":        {"hi": "महिला",    "en": "Women"},
    "cat_health":       {"hi": "स्वास्थ्य","en": "Health"},
    "tab_text":         {"hi": "⌨️  टेक्स्ट से पूछें",  "en": "⌨️  Ask by Text"},
    "tab_voice":        {"hi": "🎤  आवाज़ से पूछें",    "en": "🎤  Ask by Voice"},
    "input_label":      {"hi": "अपनी ज़रूरत लिखें",     "en": "Describe your need"},
    "input_sub":        {"hi": "हिंदी, हिंग्लिश या English — तीनों चलेगी",
                         "en": "Hindi, Hinglish or English — all accepted"},
    "placeholder":      {"hi": "जैसे: मुझे पढ़ाई के लिए पैसे चाहिए  |  kisan loan chahiye  |  I need a job",
                         "en": "e.g., I need money for studies  |  kisan loan chahiye  |  health scheme batao"},
    "search_btn":       {"hi": "🔍  योजना खोजें",  "en": "🔍  Find Scheme"},
    "clear_btn":        {"hi": "🗑  साफ करें",      "en": "🗑  Clear"},
    "found_label":      {"hi": "✅ योजनाएं मिलीं",  "en": "✅ Schemes Found"},
    "confidence_label": {"hi": "🤖 AI विश्वास स्तर", "en": "🤖 AI Confidence"},
    "open_hindi":       {"hi": "🔗 हिंदी वेबसाइट खोलें", "en": "🔗 Open Hindi Website"},
    "open_english":     {"hi": "🌐 English वेबसाइट खोलें","en": "🌐 Open English Website"},
    "warn_empty":       {"hi": "⚠️ कृपया पहले अपनी ज़रूरत लिखें।",
                         "en": "⚠️ Please describe your need first."},
    "voice_title":      {"hi": "आवाज़ से पूछें",   "en": "Ask by Voice"},
    "voice_sub":        {"hi": "नीचे दिए बटन को दबाएं और अपनी ज़रूरत हिंदी में बोलें।\nAI आपकी आवाज़ सुनकर सही योजना बताएगा।",
                         "en": "Press the button below and speak your need in Hindi or English.\nAI will listen and find the right scheme for you."},
    "voice_heard":      {"hi": "🎤 आपने बोला:", "en": "🎤 You said:"},
    "voice_searching":  {"hi": "🔄 योजनाएं ढूंढ रहे हैं…", "en": "🔄 Searching schemes…"},
    "voice_no_text":    {"hi": "⚠️ आवाज़ नहीं पहचानी गई। फिर से कोशिश करें।",
                         "en": "⚠️ Voice not recognised. Please try again."},
    "examples_label":   {"hi": "📝 उदाहरण सवाल — क्लिक करके देखें",
                         "en": "📝 Example Questions — Click to Try"},
    "model_label":      {"hi": "🤖 मॉडल जानकारी",   "en": "🤖 Model Info"},
    "accuracy_key":     {"hi": "सटीकता",             "en": "Accuracy"},
    "categories_key":   {"hi": "श्रेणियाँ",           "en": "Categories"},
    "languages_key":    {"hi": "भाषाएं",              "en": "Languages"},
    "algo_key":         {"hi": "एल्गोरिदम",           "en": "Algorithm"},
    "algo_val":         {"hi": "TF-IDF + LR",         "en": "TF-IDF + Logistic Reg."},
    "cat_label":        {"hi": "📂 श्रेणियाँ",         "en": "📂 Categories"},
    "helpline_label":   {"hi": "📞 हेल्पलाइन",        "en": "📞 Helplines"},
    "history_label":    {"hi": "🕐 हाल की खोजें",     "en": "🕐 Recent Searches"},
    "clear_hist":       {"hi": "🗑  इतिहास साफ करें", "en": "🗑  Clear History"},
    "csc_tip":          {"hi": "💡 नज़दीकी <strong>CSC</strong> या <strong>ग्राम पंचायत</strong> जाएं",
                         "en": "💡 Visit your nearest <strong>CSC</strong> or <strong>Gram Panchayat</strong>"},
    "footer_brand":     {"hi": "🇮🇳 सरकारी योजना सहायक — ग्रामीण भारत के लिए",
                         "en": "🇮🇳 Government Scheme Finder — For Rural India"},
    "footer_made":      {"hi": "Built with ❤️ for Rural India",
                         "en": "Built with ❤️ for Rural India"},
    "toggle_to_en":     {"hi": "🌐 Switch to English", "en": "🌐 हिंदी में देखें"},
    "lang_chips":       {"hi": ["🇮🇳 हिंदी", "🗣 हिंग्लिश", "🌐 English"],
                         "en": ["🇮🇳 Hindi",  "🗣 Hinglish", "🌐 English"]},
    "feedback_q":       {"hi": "क्या यह जानकारी उपयोगी थी?", "en": "Was this result helpful?"},
    "feedback_yes":     {"hi": "👍 हाँ",   "en": "👍 Yes"},
    "feedback_no":      {"hi": "👎 नहीं", "en": "👎 No"},
    "feedback_thanks":  {"hi": "✅ धन्यवाद! आपकी प्रतिक्रिया मिल गई।",
                         "en": "✅ Thank you for your feedback!"},
    "start_voice":      {"hi": "🎙️  बोलना शुरू करें", "en": "🎙️  Start Speaking"},
    "stop_voice":       {"hi": "⏹️  रोकें",            "en": "⏹️  Stop"},
    "searching":        {"hi": "🔄 योजनाएं ढूंढ रहे हैं…", "en": "🔄 Finding schemes…"},
}

def t(key):
    """Return UI string for current language."""
    return UI[key][LANG]


# ══════════════════════════════════════════════════════════════════
# SCHEME DATABASE  — each scheme has:
#   name_hi / name_en   : display names
#   elig_hi / elig_en   : eligibility text
#   url_hi              : Google-translated Hindi URL
#   url_en              : direct English URL
# ══════════════════════════════════════════════════════════════════
GT = "https://translate.google.com/translate?sl=en&tl=hi&u="

schemes = {
    "scholarship": {
        "label_hi": "📚 शिक्षा / छात्रवृत्ति",
        "label_en": "📚 Education / Scholarship",
        "icon": "📚",
        "schemes": [
            {
                "name_hi":  "📚 नेशनल स्कॉलरशिप पोर्टल (NSP)",
                "name_en":  "📚 National Scholarship Portal (NSP)",
                "elig_hi":  "✅ पात्रता: कक्षा 1 से PG | परिवार की आय ₹2.5 लाख से कम",
                "elig_en":  "✅ Eligibility: Class 1 to PG | Family income below ₹2.5 lakh",
                "url_hi":   GT + "scholarships.gov.in",
                "url_en":   "https://scholarships.gov.in",
            },
            {
                "name_hi":  "📚 पोस्ट मैट्रिक छात्रवृत्ति",
                "name_en":  "📚 Post Matric Scholarship",
                "elig_hi":  "✅ पात्रता: SC/ST/OBC छात्र | कक्षा 11 के बाद",
                "elig_en":  "✅ Eligibility: SC/ST/OBC students | After Class 11",
                "url_hi":   GT + "socialjustice.gov.in",
                "url_en":   "https://socialjustice.gov.in",
            },
            {
                "name_hi":  "📚 PM यशस्वी योजना — ₹75,000–₹1,25,000",
                "name_en":  "📚 PM YASASVI Scheme — ₹75,000–₹1,25,000",
                "elig_hi":  "✅ पात्रता: OBC/EBC/DNT छात्र | कक्षा 9 या 11",
                "elig_en":  "✅ Eligibility: OBC/EBC/DNT students | Class 9 or 11",
                "url_hi":   GT + "yet.nta.ac.in",
                "url_en":   "https://yet.nta.ac.in",
            },
            {
                "name_hi":  "📚 विद्या लक्ष्मी — एजुकेशन लोन",
                "name_en":  "📚 Vidya Lakshmi — Education Loan",
                "elig_hi":  "✅ पात्रता: कोई भी छात्र जिसे लोन चाहिए",
                "elig_en":  "✅ Eligibility: Any student needing an education loan",
                "url_hi":   GT + "vidyalakshmi.co.in",
                "url_en":   "https://www.vidyalakshmi.co.in",
            },
            {
                "name_hi":  "📚 केंद्रीय क्षेत्र छात्रवृत्ति",
                "name_en":  "📚 Central Sector Scholarship",
                "elig_hi":  "✅ पात्रता: 12वीं में 80%+ | परिवार की आय ₹4.5 लाख से कम",
                "elig_en":  "✅ Eligibility: 80%+ in 12th | Family income below ₹4.5 lakh",
                "url_hi":   GT + "scholarships.gov.in",
                "url_en":   "https://scholarships.gov.in",
            },
        ]
    },
    "agriculture": {
        "label_hi": "🌾 खेती / किसान",
        "label_en": "🌾 Agriculture / Farmers",
        "icon": "🌾",
        "schemes": [
            {
                "name_hi":  "🌾 PM किसान सम्मान निधि — हर साल ₹6,000",
                "name_en":  "🌾 PM Kisan Samman Nidhi — ₹6,000/year",
                "elig_hi":  "✅ पात्रता: सभी किसान जिनके पास खेती की ज़मीन है",
                "elig_en":  "✅ Eligibility: All farmers with cultivable land",
                "url_hi":   GT + "pmkisan.gov.in",
                "url_en":   "https://pmkisan.gov.in",
            },
            {
                "name_hi":  "🌾 PM फसल बीमा योजना",
                "name_en":  "🌾 PM Fasal Bima Yojana (Crop Insurance)",
                "elig_hi":  "✅ पात्रता: सभी किसान | फसल खराब होने पर मुआवजा",
                "elig_en":  "✅ Eligibility: All farmers | Compensation for crop loss",
                "url_hi":   GT + "pmfby.gov.in",
                "url_en":   "https://pmfby.gov.in",
            },
            {
                "name_hi":  "🌾 किसान क्रेडिट कार्ड (KCC)",
                "name_en":  "🌾 Kisan Credit Card (KCC)",
                "elig_hi":  "✅ पात्रता: सभी किसान | 4% ब्याज दर पर लोन",
                "elig_en":  "✅ Eligibility: All farmers | Loan at 4% interest",
                "url_hi":   GT + "pmkisan.gov.in",
                "url_en":   "https://pmkisan.gov.in",
            },
            {
                "name_hi":  "🌾 PM कृषि सिंचाई योजना",
                "name_en":  "🌾 PM Krishi Sinchai Yojana",
                "elig_hi":  "✅ पात्रता: सभी किसान | सिंचाई उपकरण पर सब्सिडी",
                "elig_en":  "✅ Eligibility: All farmers | Subsidy on irrigation equipment",
                "url_hi":   GT + "pmksy.gov.in",
                "url_en":   "https://pmksy.gov.in",
            },
            {
                "name_hi":  "🌾 सॉइल हेल्थ कार्ड — मुफ्त मिट्टी जाँच",
                "name_en":  "🌾 Soil Health Card — Free Soil Testing",
                "elig_hi":  "✅ पात्रता: सभी किसान | हर 2 साल में मुफ्त",
                "elig_en":  "✅ Eligibility: All farmers | Free every 2 years",
                "url_hi":   GT + "soilhealth.dac.gov.in",
                "url_en":   "https://soilhealth.dac.gov.in",
            },
        ]
    },
    "employment": {
        "label_hi": "💼 रोजगार / नौकरी",
        "label_en": "💼 Employment / Jobs",
        "icon": "💼",
        "schemes": [
            {
                "name_hi":  "💼 मनरेगा — 100 दिन काम की गारंटी",
                "name_en":  "💼 MGNREGS — 100 Days Work Guarantee",
                "elig_hi":  "✅ पात्रता: ग्रामीण क्षेत्र के वयस्क | जॉब कार्ड ज़रूरी",
                "elig_en":  "✅ Eligibility: Rural adults | Job card required",
                "url_hi":   GT + "nrega.nic.in",
                "url_en":   "https://nrega.nic.in",
            },
            {
                "name_hi":  "💼 PM कौशल विकास योजना — मुफ्त ट्रेनिंग",
                "name_en":  "💼 PM Kaushal Vikas Yojana — Free Training",
                "elig_hi":  "✅ पात्रता: 15–45 साल के युवा | 10वीं पास या ड्रॉपआउट",
                "elig_en":  "✅ Eligibility: Youth 15–45 yrs | 10th pass or dropout",
                "url_hi":   GT + "pmkvyofficial.org",
                "url_en":   "https://www.pmkvyofficial.org",
            },
            {
                "name_hi":  "💼 DDU-GKY — ग्रामीण युवाओं के लिए",
                "name_en":  "💼 DDU-GKY — Rural Youth Employment",
                "elig_hi":  "✅ पात्रता: 15–35 साल | BPL परिवार को प्राथमिकता",
                "elig_en":  "✅ Eligibility: Age 15–35 | BPL families prioritised",
                "url_hi":   GT + "ddugky.gov.in",
                "url_en":   "https://ddugky.gov.in",
            },
            {
                "name_hi":  "💼 PM रोजगार सृजन कार्यक्रम (PMEGP)",
                "name_en":  "💼 PM Employment Generation Programme (PMEGP)",
                "elig_hi":  "✅ पात्रता: 18+ साल | खुद का व्यवसाय के लिए ₹25 लाख तक",
                "elig_en":  "✅ Eligibility: 18+ yrs | Up to ₹25 lakh for own business",
                "url_hi":   GT + "kviconline.gov.in",
                "url_en":   "https://www.kviconline.gov.in",
            },
            {
                "name_hi":  "💼 नेशनल करियर सर्विस पोर्टल",
                "name_en":  "💼 National Career Service Portal",
                "elig_hi":  "✅ पात्रता: सभी नौकरी ढूंढने वाले | मुफ्त रजिस्ट्रेशन",
                "elig_en":  "✅ Eligibility: All job seekers | Free registration",
                "url_hi":   GT + "ncs.gov.in",
                "url_en":   "https://www.ncs.gov.in",
            },
        ]
    },
    "women_support": {
        "label_hi": "👩 महिला सशक्तिकरण",
        "label_en": "👩 Women Empowerment",
        "icon": "👩",
        "schemes": [
            {
                "name_hi":  "👩 बेटी बचाओ बेटी पढ़ाओ",
                "name_en":  "👩 Beti Bachao Beti Padhao",
                "elig_hi":  "✅ पात्रता: 0–10 साल की बेटियाँ | सभी परिवार",
                "elig_en":  "✅ Eligibility: Daughters aged 0–10 | All families",
                "url_hi":   GT + "wcd.nic.in",
                "url_en":   "https://wcd.nic.in",
            },
            {
                "name_hi":  "👩 PM उज्ज्वला योजना — मुफ्त गैस कनेक्शन",
                "name_en":  "👩 PM Ujjwala Yojana — Free LPG Connection",
                "elig_hi":  "✅ पात्रता: BPL परिवार की महिलाएं | APL भी पात्र",
                "elig_en":  "✅ Eligibility: BPL women | APL families also eligible",
                "url_hi":   GT + "pmuy.gov.in",
                "url_en":   "https://www.pmuy.gov.in",
            },
            {
                "name_hi":  "👩 PM मातृत्व वंदना योजना — ₹5,000",
                "name_en":  "👩 PM Matritva Vandana Yojana — ₹5,000",
                "elig_hi":  "✅ पात्रता: गर्भवती व स्तनपान कराने वाली महिलाएं",
                "elig_en":  "✅ Eligibility: Pregnant & lactating women",
                "url_hi":   GT + "pmmvy.wcd.gov.in",
                "url_en":   "https://pmmvy.wcd.gov.in",
            },
            {
                "name_hi":  "👩 महिला शक्ति केंद्र",
                "name_en":  "👩 Mahila Shakti Kendra",
                "elig_hi":  "✅ पात्रता: सभी ग्रामीण महिलाएं | कौशल व रोजगार सहायता",
                "elig_en":  "✅ Eligibility: All rural women | Skill & employment support",
                "url_hi":   GT + "wcd.nic.in",
                "url_en":   "https://wcd.nic.in",
            },
            {
                "name_hi":  "👩 स्वयं सहायता समूह (SHG) ऋण",
                "name_en":  "👩 Self Help Group (SHG) Loan",
                "elig_hi":  "✅ पात्रता: महिलाएं | SHG के माध्यम से सस्ती ब्याज दर पर ऋण",
                "elig_en":  "✅ Eligibility: Women | Low-interest loan via SHG",
                "url_hi":   GT + "nrlm.gov.in",
                "url_en":   "https://nrlm.gov.in",
            },
        ]
    },
    "health": {
        "label_hi": "🏥 स्वास्थ्य / इलाज",
        "label_en": "🏥 Health / Treatment",
        "icon": "🏥",
        "schemes": [
            {
                "name_hi":  "🏥 आयुष्मान भारत — ₹5 लाख तक मुफ्त इलाज",
                "name_en":  "🏥 Ayushman Bharat — Free treatment up to ₹5 lakh",
                "elig_hi":  "✅ पात्रता: SECC 2011 सूची में नाम | BPL परिवार",
                "elig_en":  "✅ Eligibility: Listed in SECC 2011 | BPL families",
                "url_hi":   GT + "pmjay.gov.in",
                "url_en":   "https://pmjay.gov.in",
            },
            {
                "name_hi":  "🏥 आयुष्मान कार्ड — यहाँ बनवाएं",
                "name_en":  "🏥 Ayushman Card — Apply Here",
                "elig_hi":  "✅ अपना नाम चेक करें और कार्ड बनवाएं",
                "elig_en":  "✅ Check your name and get the card issued",
                "url_hi":   GT + "beneficiary.nha.gov.in",
                "url_en":   "https://beneficiary.nha.gov.in",
            },
            {
                "name_hi":  "🏥 राष्ट्रीय स्वास्थ्य बीमा योजना (RSBY)",
                "name_en":  "🏥 Rashtriya Swasthya Bima Yojana (RSBY)",
                "elig_hi":  "✅ पात्रता: BPL परिवार | ₹30,000 तक बीमा",
                "elig_en":  "✅ Eligibility: BPL families | Insurance up to ₹30,000",
                "url_hi":   GT + "rsby.gov.in",
                "url_en":   "https://rsby.gov.in",
            },
            {
                "name_hi":  "🏥 जननी सुरक्षा योजना",
                "name_en":  "🏥 Janani Suraksha Yojana (Maternal Safety)",
                "elig_hi":  "✅ पात्रता: गर्भवती महिलाएं | BPL / SC / ST",
                "elig_en":  "✅ Eligibility: Pregnant women | BPL / SC / ST",
                "url_hi":   GT + "nhm.gov.in",
                "url_en":   "https://nhm.gov.in",
            },
            {
                "name_hi":  "🏥 Helpline: Ayushman 📞 14555 | NHM 📞 1800-180-1104",
                "name_en":  "🏥 Helpline: Ayushman 📞 14555 | NHM 📞 1800-180-1104",
                "elig_hi":  "✅ किसी भी सवाल के लिए — बिल्कुल मुफ्त",
                "elig_en":  "✅ For any health query — completely free",
                "url_hi":   None,
                "url_en":   None,
            },
        ]
    }
}


# ══════════════════════════════════════════════════════════════════
# TRAINING DATA + MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    data = {
        "text": [
            # ── SCHOLARSHIP / EDUCATION (132) ──────────────────────────────
            "I need scholarship","money for studies","financial help for education","student scholarship scheme",
            "fee support for college","I cannot pay my fees","education funding needed","scholarship for poor students",
            "fees nahi bhar pa raha","study ke paise nahi hai","college fees bahut high hai help chahiye",
            "mujhe padhai ke liye paise chahiye","mujhe fees bharne mein madad chahiye","chhatravritti chahiye",
            "education loan ya scholarship chahiye","books ke liye paise chahiye","hostel fees ke liye help",
            "student hu paise nahi hai","garib chhatra ke liye yojana","school fees bhar nahi pa raha",
            "college ke liye financial help","scholarship kaise milegi","need help for studies urgently",
            "padhai ruk jayegi paise nahi hai","tuition fees help chahiye","govt scholarship batao",
            "education support scheme batao","meri fees pending hai help chahiye","study ke liye fund chahiye",
            "mere pass apni beti ko padhane ke paise nahi hai","student scholarship apply kaise kare",
            "education ke liye govt help chahiye","meri padhai ke liye support chahiye",
            "scholarship eligibility kya hai","higher studies ke liye fund chahiye",
            "meri fees due hai kya help milegi","education ke liye loan ya grant chahiye",
            "गरीब छात्र के लिए scholarship","padhai ke liye koi yojana batao",
            "college admission ke liye paisa chahiye","student ke liye govt aid",
            "free education scheme hai kya","study ke liye paise arrange nahi ho rahe",
            "scholarship form kaha milega","education ke liye financial support",
            "mujhe padhne ke liye madad chahiye","fees waiver scheme batao",
            "school ke liye scholarship chahiye","meri education continue karni hai help karo",
            "student ke liye paisa kaise milega","mujhe apni beti ko padhane ke liye paise chahiye",
            "meri fees bharne ke liye koi scheme hai kya","padhai ke liye loan lena padega kya",
            "govt student ke liye kya help deti hai","education ke liye paisa arrange nahi ho raha",
            "scholarship milne ka process kya hai","12th ke baad scholarship milegi kya",
            "college admission ke liye fund nahi hai","गरीब छात्रों के लिए कौन सी योजना है",
            "mujhe engineering ke liye paisa chahiye","higher education ke liye loan chahiye",
            "meri padhai beech me ruk jayegi help karo","scholarship ke liye apply kaha kare",
            "govt se padhai ke liye madad chahiye","meri beti ki padhai ke liye paisa chahiye",
            "hostel aur mess fees ke liye help chahiye","student hu earning nahi hai help chahiye",
            "padhai ke liye koi govt support hai kya","scholarship ke liye documents kya chahiye",
            "education ke liye free scheme hai kya","school se college tak support milega kya",
            "meri padhai continue karne ke liye madad karo","study loan lena safe hai kya",
            "govt scholarship kitni milti hai","padhai ke liye paisa kaha se milega",
            "scholarship late aati hai kya","meri fees pending hai kya govt help karegi",
            "study ke liye sponsor chahiye","financial problem ki wajah se padhai ruk gayi",
            "education ke liye govt scheme batao detail me",
            "scholarship apply karna hai","need scholarship urgently","fees bharne ke liye help chahiye",
            "student hu financial problem hai","college fees afford nahi ho rahi","education ke liye funding chahiye",
            "govt scholarship list batao","how to get scholarship in india","fees ke liye support chahiye",
            "study ke liye paisa nahi hai","mujhe scholarship mil sakti hai kya","poor student ke liye scheme",
            "loan ya scholarship better kya hai","education ke liye sponsor chahiye",
            "padhai ke liye paisa arrange nahi ho raha",
            "govt education support details","scholarship ka result kab aata hai",
            "fees waiver ka process kya hai","scholarship ke liye documents",
            "higher studies ke liye fund","study continue nahi ho pa rahi",
            "urgent education help chahiye","scholarship ke liye eligibility",
            "college ke liye paisa arrange nahi ho raha",
            "kya mujhe govt help milegi padhai ke liye","need help for education","student financial aid chahiye",
            "scholarship portal batao","education ke liye grant chahiye",
            "free education scheme kaise milegi","padhai ke liye loan lena hai",
            "meri fees due hai help karo","student ke liye best scheme",
            "scholarship ke liye apply link","govt aid for students","mujhe education support chahiye",
            "scholarship kaise check kare","study ke liye help urgently",
            "education fund kaise milega","mujhe padhai ke liye paise chahiye",

            # ── AGRICULTURE / KISAN (122) ─────────────────────────────────
            "farmer loan scheme","crop subsidy","agriculture support scheme","kisan yojana",
            "farmer needs money","crop damage help","subsidy for seeds","kheti ke liye paise chahiye",
            "fasal kharab ho gayi","kisan loan chahiye","beej ke liye paise chahiye",
            "tractor ke liye loan chahiye","kisan ke liye yojana","kheti ke liye paisa chahiye",
            "fasal kharab ho gayi madad chahiye","kisanon ke liye sahayata","kheti me nuksaan ho gaya",
            "crop insurance chahiye","pm kisan scheme kya hai","fasal barbaad ho gayi",
            "kisan ko subsidy milegi kya","irrigation ke liye help","kheti ke liye loan",
            "seed subsidy chahiye","agriculture scheme batao","farming support chahiye",
            "kisan ke liye govt help","crop loss compensation","soil improvement scheme",
            "kisan credit card kaise banega","farming ke liye irrigation support",
            "drip irrigation scheme hai kya","agriculture loan interest kitna hai",
            "kheti ke liye machine subsidy","harvest ke baad loss ho gaya help",
            "fertilizer subsidy kaise milegi","soil testing scheme kya hai",
            "organic farming ke liye support","kisan ke liye insurance details",
            "crop ke liye protection scheme","farming ke liye govt training",
            "agriculture equipment loan chahiye","kheti ke liye pani ki problem hai help",
            "weather se fasal kharab ho gayi help","kisan ke liye best yojana batao",
            "farming me profit kaise badhaye help","govt support for small farmers",
            "kisan ke liye financial aid","kheti me profit nahi ho raha help chahiye",
            "govt farming ke liye kya support deti hai","kisan ke liye loan ka process kya hai",
            "fasal barbad ho gayi insurance milega kya","crop ke liye best subsidy kaunsi hai",
            "small farmer ke liye scheme batao","kheti ke liye pani ka issue hai kya kare",
            "tractor kharidne ke liye loan chahiye","govt se kisan ko kitna paisa milta hai",
            "fertilizer mehenga hai subsidy milegi kya","kisan credit card ka benefit kya hai",
            "agriculture me loss ho gaya help chahiye","kheti ke liye modern tools chahiye",
            "govt farming training deti hai kya","crop protection ke liye scheme hai kya",
            "beej aur khaad ke liye loan chahiye","farming ke liye govt grant hai kya",
            "kisan ke liye monthly income scheme","fasal bechne me problem aa rahi hai help",
            "mandi rate ka issue hai kya kare","govt se irrigation ke liye support milega kya",
            "soil health card kya hota hai","organic farming ke liye scheme batao",
            "agriculture me technology use karne ke liye help","kisan ke liye best govt plan kaunsa hai",
            "crop ke liye weather protection scheme","farming business start karna hai help",
            "govt se kisan ko subsidy kaise milti hai","kheti ke liye paise kaha se milege",
            "agriculture ke liye free training hai kya","kisan ke liye naye govt schemes kya hai",
            "fasal ke liye protection ka best tarika kya hai",
            "kisan loan apply kaise kare","crop insurance kaise milega",
            "farming ke liye subsidy chahiye","kheti me loss ho gaya",
            "irrigation system chahiye","seed subsidy ka process",
            "tractor loan kaise milega","agriculture support chahiye",
            "kisan credit card apply","govt farming scheme batao",
            "farming ke liye loan interest","crop damage compensation",
            "kheti ke liye paani problem","organic farming support",
            "fertilizer subsidy chahiye","small farmer ke liye help",
            "govt se kisan ko kya milta hai","fasal barbad ho gayi help karo",
            "agriculture training chahiye","farming ke liye grant",
            "kisan ke liye best scheme","soil testing kaise kare",
            "drip irrigation scheme","kisan ke liye monthly income",
            "farming business start karna hai","mandi rate problem hai",
            "crop protection kaise kare","govt subsidy kaise milegi",
            "kisan ke liye loan process","agriculture me profit kaise badhaye",
            "farming tools ke liye help","beej aur khaad ke liye paisa",
            "agriculture scheme details","kisan ke liye financial aid kisan",
            "govt farming support kya hai","crop ke liye insurance chahiye",
            "irrigation ke liye loan","kheti ke liye modern tools",
            "kisan ke liye new schemes","kisan sambandhit scheme",

            # ── EMPLOYMENT / ROJGAR (122) ─────────────────────────────────
            "I need a job","job for unemployed","employment scheme","work opportunities",
            "job assistance","unemployment help","looking for work","need work urgently",
            "mujhe naukri chahiye","job dilado","kaam chahiye bhai","koi kaam milega kya",
            "naukri nahi mil rahi","berozgar hoon madad chahiye","kaam chahiye",
            "rojgar yojana batao","job nahi mil rahi help karo","part time job chahiye",
            "freshers ke liye job","skill development scheme batao","kaam nahi mil raha",
            "income source chahiye","self employment scheme","startup help chahiye",
            "govt job scheme","daily wage kaam chahiye","training ke baad job milegi kya",
            "skill seekh ke job chahiye","rojgar ke liye help","mujhe rozgaar chahiye",
            "job ke liye training chahiye","government job kaise milegi",
            "online job opportunities batao","skill development ka course chahiye",
            "earning ka source chahiye","work from home job chahiye",
            "job ke liye resume kaise banaye","rojgar ke liye registration kaise kare",
            "nayi job ke liye apply kaise kare","career start karne ke liye help",
            "internship chahiye freshers ke liye","skill training free hai kya",
            "job ke liye govt portal batao","self business start karna hai help",
            "income badhane ke liye kya kare","job ke liye guidance chahiye",
            "rojgar ka form kaha milega","daily income ke liye kaam chahiye",
            "job ke liye financial support","kaam ke liye training chahiye",
            "job ke liye kaha apply kare","mujhe ghar ke paas job chahiye",
            "govt job ke liye kya process hai","private job nahi mil rahi kya kare",
            "online earning kaise kare","work from home ka option hai kya",
            "skill sikh ke job milegi kya","govt free training deti hai kya",
            "rojgar ke liye best scheme batao","job ke liye resume strong kaise banaye",
            "mujhe part time earning chahiye","ghar baithe kaam mil sakta hai kya",
            "startup ke liye fund kaise milega","business ke liye loan lena hai",
            "job ke liye qualification kam hai help","kaam ke liye experience nahi hai kya kare",
            "mujhe daily income chahiye urgently","job ke liye placement help chahiye",
            "govt rojgar mela kya hota hai","skill india program kaise join kare",
            "job ke liye guidance kaun dega","career kaise choose kare help",
            "mujhe stable income chahiye","freelancing kaise start kare",
            "online job safe hai kya","govt job ke liye coaching free hai kya",
            "training ke baad placement milega kya","job ke liye registration kaha kare",
            "employment exchange kya hota hai","mujhe apna kaam start karna hai help",
            "job chahiye urgently","kaam nahi mil raha","rojgar scheme batao",
            "need job support","part time job chahiye","work from home job",
            "job ke liye apply kaise kare","fresher job chahiye",
            "skill training chahiye","online job opportunities",
            "income source chahiye","resume kaise banaye",
            "job ke liye guidance","govt job ka process",
            "private job nahi mil rahi","earning kaise start kare",
            "freelancing kaise kare","internship chahiye",
            "startup help chahiye","business start karna hai",
            "rojgar registration kaise kare","placement help chahiye",
            "job portal batao","skill india program join",
            "ghar baithe job milegi kya","daily income ka kaam",
            "job ke liye qualification kam hai","career kaise choose kare",
            "job ke liye training","govt rojgar mela",
            "online earning safe hai kya","job ke liye best option",
            "earning badhane ke tareeke","job ke liye support",
            "kaam ke liye experience nahi hai","resume strong kaise kare",
            "stable income chahiye","job ke liye help urgently",
            "employment support chahiye","mujhe job chahiye",

            # ── WOMEN SUPPORT / MAHILA (122) ──────────────────────────────
            "help for women","women empowerment scheme","support for girls","scheme for women safety",
            "financial help for women","mahila yojana","ladkiyon ke liye yojana",
            "women ke liye loan hai kya","mahila ke liye sarkari help","beti bachao scheme",
            "mahila ko naukri dilao","pregnant women help","mahila ka ilaj chahiye",
            "ladki ki padhai ke liye help","women safety scheme","mahila ke liye free training",
            "beti ke liye yojana","govt ladies ke liye kya scheme hai","women ke liye koi yojana",
            "mahila ko paisa chahiye","ladki ke liye education help","widow support scheme",
            "vidhwa pension scheme","mahila ke liye business loan","women self help group",
            "SHG loan chahiye","mahila ke liye koi govt scheme","women ke liye sarkari yojana",
            "beti ka vivah ke liye madad","shadi ke liye help chahiye","mahila ke liye job scheme",
            "pregnant ke liye free hospital","mahila ke liye free medicine","ladki ka support chahiye",
            "aangan wadi scheme","mahila ke liye poshan scheme","nutrition support for women",
            "महिला के लिए योजना","बेटी बचाओ बेटी पढ़ाओ","ujjwala yojana kya hai",
            "free gas connection chahiye","mahila ke liye ujjwala scheme",
            "women farmer ke liye scheme","ladki ko scholarship chahiye",
            "mahila ke liye koi training programme","domestic violence help chahiye",
            "mahila ke liye free legal help","women helpline number kya hai",
            "beti ke rishte ke liye govt help","ladki ki shadi ke liye fund",
            "mahila entrepreneur ke liye loan","women ke liye startup fund",
            "mahila ko aarthik madad chahiye","govt se mahila ko kya milta hai",
            "mahila suraksha yojana","safe city scheme for women",
            "anganwadi worker ke liye scheme","mahila ke liye pension",
            "vidhwa pension kaise milegi","single mother ke liye help",
            "mahila ke liye housing scheme","ladki ke liye free hostel",
            "women ke liye health checkup free","mahila ke liye ration card",
            "mahila ke liye free silai training","women ke liye skill development",
            "mahila ke liye bank account scheme","jan dhan yojana for women",
            "mahila ke liye microfinance","mahila ke liye kisan scheme",
            "women ke liye govt grant","mahila ke liye insurance scheme",
            "ladki ke liye best sarkari scheme","beti ke liye govt help detail me",
            "mahila ke liye free education scheme kya hai",
            "working women ke liye govt support kya hai",
            "mahila self employment ke liye kya help milti hai",
            "ladkiyon ke liye safety apps ya schemes kya hai",
            "women ke liye financial independence kaise milegi",
            "help for women","women support chahiye","mahila loan scheme",
            "beti ke liye scholarship","widow pension kaise milegi",
            "women ke liye job scheme","mahila ke liye business loan",
            "ladki ke liye education help","women safety scheme",
            "ujjwala yojana details","free gas connection chahiye",
            "mahila ke liye training","women entrepreneur support",
            "mahila ke liye govt aid","ladki ki shaadi ke liye help",
            "women ke liye startup fund","mahila ke liye insurance",
            "anganwadi scheme kya hai","mahila ke liye pension",
            "single mother help","women ke liye health scheme",
            "mahila ke liye ration card","women self help group loan",
            "mahila ke liye microfinance","ladkiyon ke liye scheme",
            "women ke liye skill training","mahila ke liye housing scheme",
            "women ke liye govt grant","mahila ke liye free education",
            "ladki ke liye hostel facility","women ke liye financial independence",
            "mahila ke liye bank scheme","jan dhan account for women",
            "women ke liye subsidy","mahila ke liye legal help",
            "domestic violence help","women helpline number",
            "mahila ke liye support urgently","working women ke liye support",
            "women ke liye govt programs","ladki ke liye best scheme",

            # ── HEALTH / SWASTHYA (132) ───────────────────────────────────
            "health support","need medical help","hospital fees are too high","no money for treatment",
            "hospital ka kharcha nahi hai","doctor ke liye paise chahiye",
            "mujhe ilaj ke liye paise chahiye","aspataal kharche mein madad chahiye",
            "swasthya yojana","ilaj ke liye sahayata","insurance scheme batao",
            "ayushman card kaise banega","free treatment scheme","medical emergency help",
            "hospital bill bahut zyada hai","operation ke liye paise chahiye",
            "health support scheme batao","garib ke ilaj ke liye yojana",
            "medicine ke paise nahi hai","treatment afford nahi kar pa raha",
            "govt hospital scheme","health card kaise milega","bimari ke liye financial help",
            "doctor fees nahi hai","emergency medical support","healthcare subsidy",
            "free surgery scheme","insurance claim help","free checkup scheme hai kya",
            "health card banwana hai kaise kare","insurance ka claim kaise kare",
            "hospital me free treatment milega kya","emergency ke liye govt help",
            "medical loan chahiye","health ke liye subsidy milegi kya",
            "doctor consultation free hai kya","operation ke liye govt scheme",
            "medicine free milegi kya","healthcare ke liye financial support",
            "bimari ke liye insurance kaise milega","health ke liye govt program",
            "treatment ke liye loan chahiye","free ambulance service hai kya",
            "serious illness ke liye help","health benefit schemes india",
            "govt hospital me free ilaj kaise milega","health scheme apply kaise kare",
            "treatment ke liye paise arrange nahi ho rahe","mujhe free treatment kaha milega",
            "govt hospital me ilaj free hai kya","insurance lena zaroori hai kya",
            "health ke liye govt kya help deti hai","emergency me paisa kaha se milega",
            "operation ke liye loan lena padega kya","medicine bahut mehengi hai help chahiye",
            "govt se health card kaise banega","serious disease ke liye scheme hai kya",
            "doctor ka kharcha kaise manage kare","health ke liye free camp kaha lagta hai",
            "govt se ilaj ke liye madad milegi kya","hospital me admission ke liye help",
            "treatment ke liye donation kaise milega","health ke liye best scheme kaunsi hai",
            "insurance claim reject ho gaya kya kare","bimari ke liye paisa arrange nahi ho raha",
            "govt hospital ka process kya hai","health ke liye free ambulance number kya hai",
            "emergency medical loan chahiye","treatment ke liye govt grant hai kya",
            "healthcare ke liye support chahiye","doctor consultation free kaha milega",
            "insurance policy ka benefit kya hai","govt health scheme kaise check kare",
            "ilaj ke liye financial help kaise milegi","health ke liye subsidy ka process kya hai",
            "hospital bill reduce kaise kare","free surgery kaha hoti hai",
            "govt health support kaise le","free health checkup kaha available hai",
            "govt hospital me treatment kaise start kare",
            "medical help chahiye","hospital bill bahut high hai","free treatment kaha milega",
            "ayushman card apply","health insurance kaise le","emergency medical help",
            "doctor fees afford nahi ho rahi","operation ke liye paisa chahiye",
            "medicine mehengi hai help","health scheme batao",
            "govt hospital free hai kya","health card kaise banega",
            "insurance claim kaise kare","free checkup scheme",
            "medical loan chahiye","treatment ke liye fund",
            "serious illness help","ambulance number kya hai",
            "healthcare subsidy chahiye","govt se ilaj ke liye help",
            "free surgery kaha hoti hai","health camp kaha lagta hai",
            "doctor consultation free","emergency ke liye paisa",
            "insurance lena zaroori hai kya","health ke liye govt support",
            "hospital admission help","treatment ke liye donation",
            "health ke liye best scheme","bimari ke liye paisa nahi hai",
            "insurance reject ho gaya","govt health scheme details",
            "ilaj ke liye loan lena hai","free ambulance service",
            "health ke liye support urgently","govt hospital ka process",
            "health ke liye subsidy","medical emergency fund",
            "healthcare ke liye help","healthcare schemes batao",
            # Critical fix: explicit "health se related" Hinglish patterns
            "health se related schemes batao","health related yojana kya hai",
            "health se judi schemes","mujhe health se related schemes chahiye",
            "health ke schemes batao","health yojana list",
            "health se related govt help","health schemes india",
            "health se related koi yojana","swasthya se related schemes",
            "health aur ilaj ke liye scheme","health related govt yojana",
            "mujhe swasthya yojana chahiye","hospital se related scheme",
            "health mein madad chahiye","medical schemes batao",
            "swasthya se related koi scheme hai kya","health care yojana",
        ],

        "intent": (
            ["scholarship"]   * 120 +
            ["agriculture"]   * 120 +
            ["employment"]    * 120 +
            ["women_support"] * 120 +
            ["health"]        * 138
        )
    }
    df = pd.DataFrame(data)

    def clean_text(text):
        text = text.lower().replace("-", " ").replace("_", " ")
        cleaned = ""
        for char in text:
            if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
                cleaned += char
        return " ".join(cleaned.split())

    df["clean_text"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
    )
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w\u0900-\u097F]+\b",
        ngram_range=(1, 3),   # trigrams help with "health se related"
        min_df=1,
        sublinear_tf=True,
        max_features=8000,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs", class_weight="balanced")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    return vectorizer, model, acc


# ══════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════
def clean_text(text):
    text = text.lower().replace("-", " ").replace("_", " ")
    cleaned = ""
    for char in text:
        if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
            cleaned += char
    return " ".join(cleaned.split())

def has_devanagari(text):
    return any("\u0900" <= ch <= "\u097F" for ch in text)

# ══════════════════════════════════════════════════════════════════
# KEYWORD OVERRIDE MAP  — explicit domain signals trump TF-IDF
# ══════════════════════════════════════════════════════════════════
KEYWORD_OVERRIDES = {
    "health": [
        "health", "swasthya", "hospital", "ilaj", "bimari", "dawai",
        "medicine", "doctor", "treatment", "surgery", "medical",
        "operation", "checkup", "ayushman", "aspataal",
        "स्वास्थ्य", "अस्पताल", "इलाज", "बीमारी", "दवाई",
    ],
    "agriculture": [
        "kisan", "kisaan", "fasal", "kheti", "farming", "farm", "crop",
        "beej", "seed", "fertilizer", "khaad", "irrigation", "sinchai",
        "pm kisan", "किसान", "फसल", "खेती", "कृषि", "बीज",
    ],
    "scholarship": [
        "scholarship", "padhai", "fees", "fee", "college", "hostel",
        "chhatravritti", "education loan", "vidya", "tuition",
        "छात्रवृत्ति", "पढ़ाई", "शिक्षा",
    ],
    "employment": [
        "naukri", "rojgar", "job", "kaam", "rozgaar", "employment",
        "internship", "skill", "berozgar", "manrega", "mgnregs",
        "नौकरी", "रोजगार", "काम",
    ],
    "women_support": [
        "mahila", "beti", "ladki", "widow", "vidhwa", "ujjwala",
        "anganwadi", "shg", "pregnant",
        "महिला", "बेटी", "लड़की", "गर्भवती",
    ],
}


# ══════════════════════════════════════════════════════════════════
# VOICE — PRE-PROCESSING BEFORE MODEL
# normalise common ASR noise and Hinglish health phrases so the
# keyword override catches them even if ASR mis-hears slightly
# ══════════════════════════════════════════════════════════════════
VOICE_NORMALISE = [
    # (pattern_substring , replacement)
    ("हेल्थ",     "health"),
    ("हेल्थ्",    "health"),
    ("हेल्ट",     "health"),
    ("health",    "health"),
    ("हॉस्पिटल", "hospital"),
    ("अस्पताल",  "hospital"),
    ("इलाज",     "ilaj"),
    ("दवाई",     "dawai"),
    ("किसान",    "kisan"),
    ("फसल",      "fasal"),
    ("खेती",     "kheti"),
    ("नौकरी",    "naukri"),
    ("रोजगार",   "rojgar"),
    ("महिला",    "mahila"),
    ("बेटी",     "beti"),
    ("पढ़ाई",    "padhai"),
    ("छात्रवृत्ति","scholarship"),
]

def normalise_voice(text: str) -> str:
    """Map common Devanagari words spoken aloud → Hinglish equivalents
    so the keyword override picks them up reliably."""
    result = text
    for pattern, replacement in VOICE_NORMALISE:
        result = result.replace(pattern, replacement)
    return result


# ══════════════════════════════════════════════════════════════════
# CHATBOT RESPONSE  (text + voice both go through here)
# ══════════════════════════════════════════════════════════════════
def chatbot_response(user_input: str, vectorizer, model):
    if not user_input.strip():
        return None, None, 0

    stripped = user_input.strip()

    # Step 1 — For voice input, normalise Devanagari → Hinglish keywords
    normalised = normalise_voice(stripped)

    # Step 2 — Language detection & translation
    try:
        is_dev = has_devanagari(normalised)
        if not is_dev:
            try:
                detected_lang = detect(normalised)
            except Exception:
                detected_lang = "hi"
            if detected_lang == "en":
                translated = GoogleTranslator(source="en", target="hi").translate(normalised)
            else:
                translated = normalised   # Hinglish — model handles natively
        else:
            translated = normalised
    except Exception:
        translated = user_input

    # Step 3 — Keyword override (works on both original + normalised)
    combined_check = (user_input + " " + normalised + " " + translated).lower()
    for intent_key, keywords in KEYWORD_OVERRIDES.items():
        for kw in keywords:
            if kw in combined_check:
                return intent_key, schemes[intent_key], 92.0

    # Step 4 — TF-IDF + Logistic Regression fallback
    cleaned = clean_text(translated)
    vec = vectorizer.transform([cleaned])
    intent = model.predict(vec)[0]
    confidence = float(model.predict_proba(vec).max()) * 100
    return intent, schemes[intent], confidence


# ══════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════
vectorizer, model, model_acc = train_model()


# ══════════════════════════════════════════════════════════════════
# CSS — Modern India Design
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');
:root {
  --saffron:#FF6B2B; --saffron-lt:#FFF0E8; --deep-green:#0F3D2B;
  --leaf:#1A5C3A; --leaf-lt:#E8F5EE; --gold:#F5A623; --gold-lt:#FFF8E6;
  --cream:#FDFAF5; --sand:#F7F0E3; --white:#FFFFFF; --charcoal:#1A1A2E;
  --body-text:#374151; --muted:#6B7280; --lighter:#9CA3AF;
  --border:#E5D9C6; --border-light:#F0E8D8; --success:#16A34A;
  --blue-pill:#EEF2FF; --blue-accent:#4F46E5;
  --shadow-sm:0 2px 8px rgba(0,0,0,0.06); --shadow-md:0 6px 24px rgba(0,0,0,0.08);
  --radius-sm:10px; --radius-md:16px; --radius-lg:24px; --radius-xl:32px;
}
html,body,[class*="css"]{font-family:'Inter','Noto Sans Devanagari',sans-serif;
  background-color:var(--cream)!important;color:var(--charcoal);}
.stApp{background:linear-gradient(145deg,#FDFAF5 0%,#F7F0E3 60%,#F0E8D8 100%)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;padding-bottom:2rem!important;max-width:1100px!important;}

/* HERO */
.hero-outer{position:relative;border-radius:var(--radius-xl);overflow:hidden;margin-bottom:1.4rem;
  box-shadow:0 16px 60px rgba(15,61,43,0.28);}
.hero-bg{background:linear-gradient(135deg,#0F3D2B 0%,#1A5C3A 45%,#2D8653 80%,#3AA876 100%);
  padding:2.6rem 2.6rem 2rem;position:relative;}
.hero-tricolor{display:flex;height:6px;}
.tc-s{flex:1;background:#FF9933;} .tc-w{flex:1;background:#FFF;} .tc-g{flex:1;background:#138808;}
.hero-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  color:rgba(255,255,255,.55);margin-bottom:.5rem;}
.hero-title{font-family:'Playfair Display',serif;font-size:2.6rem;font-weight:800;
  color:#FFF;margin:0 0 .3rem;line-height:1.1;}
.hero-title .accent{color:var(--gold);}
.hero-sub{color:rgba(255,255,255,.7);font-size:.95rem;margin:0 0 1.4rem;max-width:520px;line-height:1.6;}
.hero-stats{display:flex;gap:1.8rem;flex-wrap:wrap;}
.hero-stat-num{font-size:1.4rem;font-weight:800;color:var(--gold);display:block;line-height:1;}
.hero-stat-label{font-size:.7rem;color:rgba(255,255,255,.55);font-weight:500;margin-top:3px;display:block;}
.hero-badges-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:1.2rem;}
.hero-badge{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  color:rgba(255,255,255,.9);padding:5px 14px;border-radius:50px;font-size:.8rem;font-weight:500;}
/* language toggle pill */
.lang-toggle-wrap{position:absolute;top:1.2rem;right:1.6rem;}
.lang-toggle-btn{background:rgba(255,255,255,.15);border:1.5px solid rgba(255,255,255,.35);
  color:#fff;padding:6px 16px;border-radius:50px;font-size:.82rem;font-weight:700;
  cursor:pointer;transition:all .2s;backdrop-filter:blur(6px);}
.lang-toggle-btn:hover{background:rgba(255,255,255,.25);}

/* CATEGORY GRID */
.cat-section-label{font-size:.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:.7rem;}
.cat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:1.4rem;}
.cat-card{background:var(--white);border:2px solid var(--border-light);border-radius:var(--radius-md);
  padding:14px 10px;text-align:center;transition:all .22s ease;box-shadow:var(--shadow-sm);}
.cat-card:hover{border-color:var(--saffron);background:var(--saffron-lt);transform:translateY(-3px);
  box-shadow:0 8px 24px rgba(255,107,43,.15);}
.cat-icon{font-size:1.5rem;display:block;margin-bottom:5px;}
.cat-name{font-size:.8rem;font-weight:600;color:var(--charcoal);line-height:1.3;}
.cat-hint{font-size:.68rem;color:var(--muted);margin-top:2px;}

/* INPUT SECTION */
.input-section{background:var(--white);border-radius:var(--radius-lg);padding:1.6rem 1.8rem;
  border:2px solid var(--border-light);box-shadow:var(--shadow-md);margin-bottom:1.4rem;
  position:relative;}
.input-section::before{content:"";position:absolute;top:0;left:0;right:0;height:4px;
  border-radius:var(--radius-lg) var(--radius-lg) 0 0;
  background:linear-gradient(90deg,#FF6B2B,#F5A623,#138808);}
.input-label-row{display:flex;align-items:center;gap:10px;margin-bottom:.8rem;}
.input-label-icon{width:32px;height:32px;background:var(--leaf-lt);border-radius:8px;
  display:flex;align-items:center;justify-content:center;font-size:1rem;}
.input-label-text{font-size:.85rem;font-weight:700;color:var(--charcoal);}
.input-label-sub{font-size:.75rem;color:var(--muted);}
.stTextInput>div>div>input{border-radius:var(--radius-md)!important;border:2px solid var(--border)!important;
  background:var(--sand)!important;padding:14px 18px!important;font-size:1.02rem!important;
  font-family:'Inter','Noto Sans Devanagari',sans-serif!important;color:var(--charcoal)!important;
  transition:all .2s!important;}
.stTextInput>div>div>input:focus{border-color:var(--leaf)!important;background:var(--white)!important;
  box-shadow:0 0 0 4px rgba(26,92,58,.1)!important;}
.stButton>button{border-radius:var(--radius-md)!important;font-weight:700!important;
  font-size:.92rem!important;padding:12px 24px!important;transition:all .22s ease!important;
  border:none!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#FF6B2B 0%,#F5A623 100%)!important;
  color:white!important;box-shadow:0 4px 18px rgba(255,107,43,.35)!important;}
.stButton>button[kind="primary"]:hover{transform:translateY(-2px)!important;
  box-shadow:0 8px 28px rgba(255,107,43,.45)!important;}
.stButton>button[kind="secondary"]{background:var(--white)!important;
  border:2px solid var(--border)!important;color:var(--body-text)!important;}

/* SCHEME CARDS */
.result-wrapper{animation:slideUp .4s cubic-bezier(.22,1,.36,1);}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.result-header{display:flex;align-items:center;gap:12px;background:var(--leaf-lt);
  border:2px solid #C3E8D4;border-radius:var(--radius-md);padding:1rem 1.4rem;margin-bottom:1rem;}
.result-icon{width:44px;height:44px;background:var(--leaf);border-radius:12px;
  display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0;}
.result-found-label{font-size:.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  color:var(--leaf);margin-bottom:2px;}
.result-category{font-size:1.05rem;font-weight:700;color:var(--deep-green);}
.scheme-card{background:var(--white);border:1.5px solid var(--border-light);
  border-left:5px solid var(--saffron);border-radius:var(--radius-md);padding:1.1rem 1.3rem;
  margin-bottom:.8rem;transition:all .22s ease;position:relative;overflow:hidden;}
.scheme-card:hover{border-left-color:var(--leaf);box-shadow:var(--shadow-md);
  transform:translateX(4px);background:#FAFFFE;}
.scheme-name{font-weight:700;font-size:.97rem;color:var(--charcoal);margin-bottom:4px;line-height:1.4;}
.scheme-eligibility{font-size:.84rem;color:var(--leaf);font-weight:500;margin-bottom:8px;line-height:1.5;}
/* dual link buttons */
.scheme-links{display:flex;gap:8px;flex-wrap:wrap;}
.scheme-link-hi a,.scheme-link-en a{display:inline-flex;align-items:center;gap:5px;
  font-size:.8rem;font-weight:700;text-decoration:none;padding:5px 12px;
  border-radius:50px;transition:all .2s;}
.scheme-link-hi a{background:var(--saffron-lt);color:var(--saffron)!important;
  border:1.5px solid rgba(255,107,43,.25);}
.scheme-link-hi a:hover{background:var(--saffron)!important;color:white!important;}
.scheme-link-en a{background:var(--blue-pill);color:var(--blue-accent)!important;
  border:1.5px solid rgba(79,70,229,.2);}
.scheme-link-en a:hover{background:var(--blue-accent)!important;color:white!important;}

/* CONFIDENCE BAR */
.confidence-wrap{background:var(--sand);border-radius:var(--radius-md);padding:1rem 1.2rem;
  margin-top:.8rem;border:1px solid var(--border-light);}
.conf-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.conf-label{font-size:.78rem;color:var(--muted);font-weight:600;}
.conf-percent{font-size:.9rem;font-weight:800;color:var(--leaf);}
.conf-bar-bg{background:var(--border-light);border-radius:50px;height:8px;overflow:hidden;}
.conf-bar-fill{height:8px;border-radius:50px;
  background:linear-gradient(90deg,#FF6B2B,#F5A623,#138808);
  transition:width 1s cubic-bezier(.22,1,.36,1);}

/* FEEDBACK */
.feedback-row{display:flex;align-items:center;gap:12px;margin-top:.8rem;
  background:var(--white);border:1.5px solid var(--border-light);border-radius:var(--radius-sm);
  padding:.7rem 1.1rem;}
.feedback-q{font-size:.82rem;color:var(--muted);font-weight:600;}

/* HELPLINE ROW */
.helpline-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:.8rem;}
.helpline-pill{background:var(--white);border:1.5px solid var(--border);border-radius:var(--radius-sm);
  padding:7px 13px;font-size:.8rem;font-weight:500;color:var(--body-text);}
.helpline-pill strong{color:var(--deep-green);}

/* EXAMPLES */
.examples-wrap{background:var(--white);border-radius:var(--radius-lg);padding:1.2rem 1.4rem;
  border:2px solid var(--border-light);box-shadow:var(--shadow-sm);margin-bottom:1.2rem;}
.ex-section-label{font-size:.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:.8rem;}
div[data-testid="column"] .stButton>button{font-size:.76rem!important;padding:6px 11px!important;
  border-radius:8px!important;font-weight:500!important;height:auto!important;
  white-space:normal!important;word-break:break-word!important;line-height:1.4!important;
  background:var(--blue-pill)!important;border:1.5px solid #C7D2FE!important;
  color:var(--blue-accent)!important;box-shadow:none!important;}
div[data-testid="column"] .stButton>button:hover{background:var(--blue-accent)!important;
  color:white!important;border-color:var(--blue-accent)!important;}

/* VOICE TAB */
.voice-panel{background:var(--white);border-radius:var(--radius-lg);padding:2rem;
  border:2px solid var(--border-light);text-align:center;}
.voice-icon-ring{width:80px;height:80px;border-radius:50%;
  background:linear-gradient(135deg,var(--leaf-lt),#C3E8D4);border:3px solid #C3E8D4;
  margin:0 auto 1rem;display:flex;align-items:center;justify-content:center;font-size:2rem;
  animation:pulse-ring 2s ease infinite;}
@keyframes pulse-ring{0%,100%{box-shadow:0 0 0 0 rgba(26,92,58,.2);}
  50%{box-shadow:0 0 0 12px rgba(26,92,58,0);}}
.voice-title{font-size:1.15rem;font-weight:700;color:var(--charcoal);margin-bottom:.3rem;}
.voice-sub{font-size:.86rem;color:var(--muted);margin-bottom:1.2rem;line-height:1.6;}
.lang-chip{background:var(--gold-lt);border:1.5px solid #F5D78A;color:#8B6400;
  border-radius:50px;padding:4px 13px;font-size:.76rem;font-weight:600;}
.voice-transcript{background:var(--leaf-lt);border:2px solid #C3E8D4;border-radius:var(--radius-md);
  padding:.9rem 1.2rem;margin:.8rem 0;font-size:.95rem;font-weight:600;color:var(--deep-green);}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--white)!important;border-radius:var(--radius-md)!important;
  padding:4px!important;border:2px solid var(--border-light)!important;gap:4px!important;margin-bottom:1rem!important;}
.stTabs [data-baseweb="tab"]{border-radius:var(--radius-sm)!important;font-weight:600!important;
  font-size:.88rem!important;color:var(--muted)!important;padding:9px 18px!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1A5C3A,#2D8653)!important;color:white!important;}

/* SIDEBAR */
.side-card{background:var(--white);border-radius:var(--radius-md);padding:1.1rem 1.3rem;
  border:2px solid var(--border-light);margin-bottom:1rem;box-shadow:var(--shadow-sm);}
.side-card-label{font-size:.7rem;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);margin-bottom:.9rem;display:flex;align-items:center;gap:6px;}
.side-card-label::after{content:"";flex:1;height:1px;background:var(--border-light);}
.model-stat-row{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid var(--border-light);font-size:.84rem;}
.model-stat-row:last-child{border-bottom:none;}
.model-stat-key{color:var(--muted);} .model-stat-val{font-weight:700;color:var(--deep-green);}
.cat-side-item{display:flex;align-items:center;gap:10px;padding:7px 0;
  border-bottom:1px dashed var(--border-light);}
.cat-side-item:last-child{border-bottom:none;}
.cat-side-icon{width:34px;height:34px;background:var(--sand);border-radius:9px;
  display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.cat-side-name{font-weight:600;font-size:.86rem;color:var(--charcoal);}
.cat-side-hint{font-size:.7rem;color:var(--lighter);margin-top:1px;}
.history-chip{background:var(--sand);border:1.5px solid var(--border-light);
  border-radius:var(--radius-sm);padding:7px 10px;margin-bottom:6px;}
.history-query{font-size:.8rem;color:var(--body-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.history-cat{font-size:.7rem;color:var(--muted);margin-top:2px;}

/* FOOTER */
.footer-wrap{background:linear-gradient(135deg,#0F3D2B 0%,#1A5C3A 100%);border-radius:var(--radius-lg);
  padding:1.1rem 1.6rem;color:white;display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:10px;font-size:.82rem;margin-top:1.2rem;}
.footer-brand{font-weight:700;}
.footer-links{display:flex;gap:1.4rem;flex-wrap:wrap;}
.footer-links a{color:var(--gold);text-decoration:none;font-weight:700;}
.footer-made{font-size:.72rem;opacity:.5;}
.footer-tribar{height:3px;border-radius:0 0 var(--radius-lg) var(--radius-lg);
  background:linear-gradient(90deg,#FF9933 33%,#FFF 33% 66%,#138808 66%);}
.stSpinner>div{border-top-color:var(--saffron)!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HERO HEADER  +  LANGUAGE TOGGLE
# ══════════════════════════════════════════════════════════════════
# Language toggle button (placed above hero so it renders inside it via CSS)
toggle_col, _ = st.columns([1, 8])
with toggle_col:
    if st.button(t("toggle_to_en"), key="lang_toggle"):
        st.session_state.lang = "en" if LANG == "hi" else "hi"
        st.rerun()

st.markdown(f"""
<div class="hero-outer">
  <div class="hero-bg">
    <div class="hero-eyebrow">{t("eyebrow")}</div>
    <div class="hero-title">{t("hero_title_1")} <span class="accent">{t("hero_title_2")}</span> {t("hero_title_3")}</div>
    <div class="hero-sub">{t("hero_sub")}</div>
    <div class="hero-stats">
      <div class="hero-stat"><span class="hero-stat-num">600+</span><span class="hero-stat-label">{t("stat_train")}</span></div>
      <div class="hero-stat"><span class="hero-stat-num">5</span><span class="hero-stat-label">{t("stat_cat")}</span></div>
      <div class="hero-stat"><span class="hero-stat-num">25+</span><span class="hero-stat-label">{t("stat_schemes")}</span></div>
      <div class="hero-stat"><span class="hero-stat-num">3</span><span class="hero-stat-label">{t("stat_lang")}</span></div>
    </div>
    <div class="hero-badges-row">
      <span class="hero-badge">📚 {t("cat_edu")}</span>
      <span class="hero-badge">🌾 {t("cat_agri")}</span>
      <span class="hero-badge">💼 {t("cat_emp")}</span>
      <span class="hero-badge">👩 {t("cat_women")}</span>
      <span class="hero-badge">🏥 {t("cat_health")}</span>
    </div>
  </div>
  <div class="hero-tricolor"><div class="tc-s"></div><div class="tc-w"></div><div class="tc-g"></div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════
col_main, col_side = st.columns([2.7, 1], gap="large")

with col_main:

    # ── CATEGORY QUICK-SELECT ──────────────────────────────────
    st.markdown(f'<div class="cat-section-label">{t("quick_select")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="cat-grid">
      <div class="cat-card"><span class="cat-icon">📚</span>
        <div class="cat-name">{t("cat_edu")}</div><div class="cat-hint">Scholarship · Fees</div></div>
      <div class="cat-card"><span class="cat-icon">🌾</span>
        <div class="cat-name">{t("cat_agri")}</div><div class="cat-hint">Kisan · Loan</div></div>
      <div class="cat-card"><span class="cat-icon">💼</span>
        <div class="cat-name">{t("cat_emp")}</div><div class="cat-hint">Job · Training</div></div>
      <div class="cat-card"><span class="cat-icon">👩</span>
        <div class="cat-name">{t("cat_women")}</div><div class="cat-hint">Beti · Mahila</div></div>
      <div class="cat-card"><span class="cat-icon">🏥</span>
        <div class="cat-name">{t("cat_health")}</div><div class="cat-hint">Ilaj · Hospital</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([t("tab_text"), t("tab_voice")])

    # ════════════════════════════════════════
    # TAB 1 — TEXT INPUT
    # ════════════════════════════════════════
    with tab1:
        st.markdown(f"""
        <div class="input-section">
          <div class="input-label-row">
            <div class="input-label-icon">✍️</div>
            <div>
              <div class="input-label-text">{t("input_label")}</div>
              <div class="input-label-sub">{t("input_sub")}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_input(
                label="query",
                label_visibility="collapsed",
               placeholder=t("placeholder"),
               value=st.session_state.query,
               key="text input" 
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            search_btn = st.button(t("search_btn"), type="primary", use_container_width=True)
        with c2:
            clear_btn = st.button(t("clear_btn"), type="secondary", use_container_width=True)

        if clear_btn:
            st.session_state.query = ""
            st.rerun()

        if search_btn and user_input.strip():
            with st.spinner(t("searching")):
                intent, scheme_data, confidence = chatbot_response(user_input, vectorizer, model)

            if intent:
                st.session_state.history.insert(0, {"query": user_input, "intent": intent})
                if len(st.session_state.history) > 8:
                    st.session_state.history = st.session_state.history[:8]

                label_key = "label_hi" if LANG == "hi" else "label_en"

                st.markdown(f"""
                <div class="result-wrapper">
                  <div class="result-header">
                    <div class="result-icon">{scheme_data['icon']}</div>
                    <div>
                      <div class="result-found-label">{t("found_label")}</div>
                      <div class="result-category">{scheme_data[label_key]}</div>
                    </div>
                  </div>
                """, unsafe_allow_html=True)

                for s in scheme_data["schemes"]:
                    name = s["name_hi"] if LANG == "hi" else s["name_en"]
                    elig = s["elig_hi"] if LANG == "hi" else s["elig_en"]
                    hi_link = ""
                    en_link = ""
                    if s["url_hi"]:
                        hi_link = f'<div class="scheme-link-hi"><a href="{s["url_hi"]}" target="_blank">🔗 {t("open_hindi")}</a></div>'
                    if s["url_en"]:
                        en_link = f'<div class="scheme-link-en"><a href="{s["url_en"]}" target="_blank">🌐 {t("open_english")}</a></div>'
                    st.markdown(f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{name}</div>
                        <div class="scheme-eligibility">{elig}</div>
                        <div class="scheme-links">{hi_link}{en_link}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence bar
                bar = int(confidence)
                st.markdown(f"""
                <div class="confidence-wrap">
                  <div class="conf-top">
                    <div class="conf-label">{t("confidence_label")}</div>
                    <div class="conf-percent">{bar:.0f}%</div>
                  </div>
                  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

                # Helpline
                st.markdown(f"""
                <div class="helpline-row">
                  <div class="helpline-pill">{t("csc_tip")}</div>
                  <div class="helpline-pill">📞 PM Kisan: <strong>1800-11-0001</strong></div>
                  <div class="helpline-pill">📞 Ayushman: <strong>14555</strong></div>
                </div>
                </div>
                """, unsafe_allow_html=True)

                # Feedback
                fb_key = f"fb_{user_input[:30]}"
                if fb_key not in st.session_state.feedback:
                    st.markdown(f'<div class="feedback-row"><span class="feedback-q">{t("feedback_q")}</span></div>',
                                unsafe_allow_html=True)
                    fc1, fc2, fc3 = st.columns([2, 1, 1])
                    with fc2:
                        if st.button(t("feedback_yes"), key=fb_key + "_y"):
                            st.session_state.feedback[fb_key] = "yes"
                            st.rerun()
                    with fc3:
                        if st.button(t("feedback_no"), key=fb_key + "_n"):
                            st.session_state.feedback[fb_key] = "no"
                            st.rerun()
                else:
                    st.success(t("feedback_thanks"))

        elif search_btn and not user_input.strip():
            st.warning(t("warn_empty"))

        # ── EXAMPLE QUESTIONS ──────────────────────────────────
        st.markdown(f"""
        <div class="examples-wrap">
          <div class="ex-section-label">{t("examples_label")}</div>
        </div>
        """, unsafe_allow_html=True)

        examples_hi = [
            "मुझे पढ़ाई के लिए पैसे चाहिए",
            "kisan loan chahiye fasal kharab ho gayi",
            "naukri nahi mil rahi help chahiye",
            "mahila ke liye koi yojana hai kya",
            "hospital ka kharcha bahut zyada hai",
            "mujhe health se related schemes batao",
            "I need a job urgently",
            "free medical treatment kaha milega",
        ]
        examples_en = [
            "I need money for studies",
            "kisan crop insurance scheme",
            "no job found help needed",
            "government scheme for women",
            "cannot afford hospital bill",
            "what are health schemes available",
            "I need a job urgently",
            "where to get free medical treatment",
        ]
        examples = examples_hi if LANG == "hi" else examples_en

        ex_cols = st.columns(4)
        for i, ex in enumerate(examples):
            with ex_cols[i % 4]:
                if st.button(ex, key=f"ex_{i}", help=ex, use_container_width=True):
                    st.session_state.query = ex
                    st.rerun()

    # ════════════════════════════════════════
    # TAB 2 — VOICE INPUT  (fixed)
    # ════════════════════════════════════════
    with tab2:
        chips_html = " ".join([f'<span class="lang-chip">{c}</span>' for c in t("lang_chips")])
        st.markdown(f"""
        <div class="voice-panel">
          <div class="voice-icon-ring">🎙️</div>
          <div class="voice-title">{t("voice_title")}</div>
          <div class="voice-sub">{t("voice_sub")}</div>
          <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;">
            {chips_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── VOICE FIX: try both hi-IN and en-IN ─────────────────
        # streamlit_mic_recorder's speech_to_text uses the Web Speech API.
        # For Hinglish accuracy we first try hi-IN; if it returns empty
        # or only punctuation we fall back to en-IN, then run the result
        # through normalise_voice() before passing to chatbot_response.
        voice_text = speech_to_text(
            language='hi-IN',
            start_prompt=t("start_voice"),
            stop_prompt=t("stop_voice"),
            just_once=True,
            use_container_width=True,
            key="mic_hi"
        )

        # If hi-IN gave nothing, offer en-IN button as fallback
        if not voice_text:
            st.caption("🇬🇧 " + ("If Hindi recognition failed, try English:" if LANG=="en"
                                   else "हिंदी नहीं पहचानी? English में बोलकर देखें:"))
            voice_text_en = speech_to_text(
                language='en-IN',
                start_prompt=("🎙️ Speak in English" if LANG=="en" else "🎙️ English में बोलें"),
                stop_prompt=t("stop_voice"),
                just_once=True,
                use_container_width=True,
                key="mic_en"
            )
            if voice_text_en:
                voice_text = voice_text_en

        if voice_text and voice_text.strip():
            # Normalise voice transcript for keyword matching
            processed = normalise_voice(voice_text.strip())
            st.markdown(f"""
            <div class="voice-transcript">
              {t("voice_heard")} <em>{voice_text}</em>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(t("voice_searching")):
                intent, scheme_data, confidence = chatbot_response(processed, vectorizer, model)

            if intent:
                label_key = "label_hi" if LANG == "hi" else "label_en"
                st.markdown(f"""
                <div class="result-header" style="margin-top:.8rem;">
                  <div class="result-icon">{scheme_data['icon']}</div>
                  <div>
                    <div class="result-found-label">{t("found_label")}</div>
                    <div class="result-category">{scheme_data[label_key]}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                for s in scheme_data["schemes"]:
                    name = s["name_hi"] if LANG == "hi" else s["name_en"]
                    elig = s["elig_hi"] if LANG == "hi" else s["elig_en"]
                    hi_link = ""
                    en_link = ""
                    if s["url_hi"]:
                        hi_link = f'<div class="scheme-link-hi"><a href="{s["url_hi"]}" target="_blank">🔗 {t("open_hindi")}</a></div>'
                    if s["url_en"]:
                        en_link = f'<div class="scheme-link-en"><a href="{s["url_en"]}" target="_blank">🌐 {t("open_english")}</a></div>'
                    st.markdown(f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{name}</div>
                        <div class="scheme-eligibility">{elig}</div>
                        <div class="scheme-links">{hi_link}{en_link}</div>
                    </div>
                    """, unsafe_allow_html=True)

                bar = int(confidence)
                st.markdown(f"""
                <div class="confidence-wrap">
                  <div class="conf-top">
                    <div class="conf-label">{t("confidence_label")}</div>
                    <div class="conf-percent">{bar:.0f}%</div>
                  </div>
                  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

                # Add to history
                st.session_state.history.insert(0, {"query": voice_text, "intent": intent})
                if len(st.session_state.history) > 8:
                    st.session_state.history = st.session_state.history[:8]

        elif voice_text is not None and not voice_text.strip():
            st.warning(t("voice_no_text"))


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with col_side:

    # Model stats
    st.markdown(f"""
    <div class="side-card">
      <div class="side-card-label">{t("model_label")}</div>
      <div class="model-stat-row">
        <span class="model-stat-key">{t("accuracy_key")}</span>
        <span class="model-stat-val">{model_acc*100:.1f}%</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">{t("categories_key")}</span>
        <span class="model-stat-val">5</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">{t("languages_key")}</span>
        <span class="model-stat-val">हिंदी · हिंग्लिश · EN</span>
      </div>
      <div class="model-stat-row">
        <span class="model-stat-key">{t("algo_key")}</span>
        <span class="model-stat-val">{t("algo_val")}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Category guide
    cat_info = [
        ("📚", t("cat_edu"),    "Scholarship, fees, study"),
        ("🌾", t("cat_agri"),   "Kisan, farming, loan"),
        ("💼", t("cat_emp"),    "Job, training, naukri"),
        ("👩", t("cat_women"),  "Women, beti, mahila"),
        ("🏥", t("cat_health"), "Hospital, ilaj, treatment"),
    ]
    st.markdown(f'<div class="side-card"><div class="side-card-label">{t("cat_label")}</div>', unsafe_allow_html=True)
    for icon, name, hint in cat_info:
        st.markdown(f"""
        <div class="cat-side-item">
          <div class="cat-side-icon">{icon}</div>
          <div><div class="cat-side-name">{name}</div><div class="cat-side-hint">{hint}</div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Helplines
    st.markdown(f"""
    <div class="side-card">
      <div class="side-card-label">{t("helpline_label")}</div>
      <div class="model-stat-row"><span class="model-stat-key">PM Kisan</span><span class="model-stat-val">1800-11-0001</span></div>
      <div class="model-stat-row"><span class="model-stat-key">Ayushman</span><span class="model-stat-val">14555</span></div>
      <div class="model-stat-row"><span class="model-stat-key">NHM</span><span class="model-stat-val">1800-180-1104</span></div>
      <div class="model-stat-row"><span class="model-stat-key">PMKVY</span><span class="model-stat-val">1800-123-9626</span></div>
      <div class="model-stat-row"><span class="model-stat-key">Women</span><span class="model-stat-val">181</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Recent history
    if st.session_state.history:
        st.markdown(f'<div class="side-card"><div class="side-card-label">{t("history_label")}</div>', unsafe_allow_html=True)
        for item in st.session_state.history[:5]:
            lk = "label_hi" if LANG == "hi" else "label_en"
            label = schemes[item["intent"]][lk]
            q = item['query'][:36] + ("…" if len(item['query']) > 36 else "")
            st.markdown(f"""
            <div class="history-chip">
              <div class="history-query">{q}</div>
              <div class="history-cat">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(t("clear_hist"), key="clear_hist_btn"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer-wrap">
  <div class="footer-brand">{t("footer_brand")}</div>
  <div class="footer-links">
    <a href="tel:18001110001">📞 PM Kisan: 1800-11-0001</a>
    <a href="tel:14555">📞 Ayushman: 14555</a>
  </div>
  <div class="footer-made">{t("footer_made")}</div>
</div>
<div class="footer-tribar"></div>
""", unsafe_allow_html=True)
