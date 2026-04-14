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


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="सरकारी योजना सहायक",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)
if "messages" not in st.session_state:
    st.session_state.messages = []
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
         st.write(message["content"])

st.title("click to speak")

text = speech_to_text(
    language='hi-IN',
    start_prompt="🎙️ Start",
    stop_prompt="⏹️ Stop",
    key="mic_test_unique"
)

st.write("Result:", text)
# ──────────────────────────────────────────────
# CUSTOM CSS — Earthy India-inspired palette
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tiro+Devanagari+Hindi&family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

/* ── Root Variables ── */
:root {
  --saffron:    #FF6B2B;
  --deep-green: #1B4332;
  --leaf:       #2D6A4F;
  --gold:       #F4A22D;
  --cream:      #FFF8EE;
  --sand:       #F5E6CC;
  --charcoal:   #1C1C1E;
  --muted:      #6B7280;
  --card-bg:    #FFFFFF;
  --border:     #E8D5B7;
  --success:    #22C55E;
  --tag-blue:   #EEF2FF;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream) !important;
    color: var(--charcoal);
}

.stApp {
    background: linear-gradient(160deg, #FFF8EE 0%, #FFF0D6 50%, #F5E6CC 100%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 960px !important; }

/* ── HERO HEADER ── */
.hero-wrap {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
    border-radius: 24px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(27,67,50,0.25);
}
.hero-wrap::before {
    content: "🇮🇳";
    position: absolute;
    font-size: 160px;
    right: -20px;
    top: -20px;
    opacity: 0.07;
    line-height: 1;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.2rem 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    color: rgba(255,255,255,0.78);
    font-size: 1rem;
    margin: 0 0 1.4rem 0;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 5px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* ── CATEGORY PILLS ── */
.cat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
}
.cat-pill {
    background: white;
    border: 2px solid var(--border);
    border-radius: 50px;
    padding: 8px 18px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--charcoal);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.cat-pill:hover {
    border-color: var(--saffron);
    background: #FFF3EC;
    color: var(--saffron);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(255,107,43,0.15);
}

/* ── INPUT AREA ── */
.input-card {
    background: white;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    border: 2px solid var(--border);
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
    margin-bottom: 1.6rem;
}
.input-label {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}

/* Streamlit text input override */
.stTextInput > div > div > input {
    border-radius: 14px !important;
    border: 2px solid var(--border) !important;
    background: var(--sand) !important;
    padding: 14px 18px !important;
    font-size: 1.05rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--charcoal) !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--leaf) !important;
    box-shadow: 0 0 0 4px rgba(45,106,79,0.1) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #A8A29E !important;
}

/* Buttons */
.stButton > button {
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B2B, #F4A22D) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(255,107,43,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255,107,43,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    border: 2px solid var(--border) !important;
    color: var(--charcoal) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--leaf) !important;
    color: var(--leaf) !important;
}

/* ── RESULT AREA ── */
.result-section {
    background: white;
    border-radius: 20px;
    padding: 1.8rem;
    border: 2px solid var(--border);
    box-shadow: 0 6px 32px rgba(0,0,0,0.07);
    margin-bottom: 1.4rem;
    animation: fadeInUp 0.4s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: var(--deep-green);
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 2px dashed var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── SCHEME CARD ── */
.scheme-card {
    background: var(--cream);
    border: 1.5px solid var(--border);
    border-left: 5px solid var(--saffron);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    transition: all 0.2s ease;
}
.scheme-card:hover {
    border-left-color: var(--leaf);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    transform: translateX(3px);
}
.scheme-name {
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--charcoal);
    margin-bottom: 4px;
}
.scheme-eligibility {
    font-size: 0.88rem;
    color: var(--leaf);
    font-weight: 500;
    margin-bottom: 6px;
}
.scheme-link a {
    color: var(--saffron) !important;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none;
}
.scheme-link a:hover { text-decoration: underline !important; }

/* ── EXAMPLE QUESTIONS ── */
.examples-section {
    background: white;
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    border: 2px solid var(--border);
    margin-bottom: 1.4rem;
}
.examples-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.ex-btn {
    display: inline-block;
    background: var(--tag-blue);
    color: #4338CA;
    border: 1px solid #C7D2FE;
    border-radius: 50px;
    padding: 6px 15px;
    font-size: 0.83rem;
    cursor: pointer;
    margin: 4px;
    font-weight: 500;
}

/* ── FOOTER ── */
.footer-strip {
    background: linear-gradient(90deg, var(--deep-green), var(--leaf));
    border-radius: 16px;
    padding: 1rem 1.5rem;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 0.85rem;
    margin-top: 1rem;
}
.footer-strip a { color: var(--gold); text-decoration: none; font-weight: 600; }

/* ── CONFIDENCE BAR ── */
.confidence-wrap { margin-top: 1rem; }
.conf-label { font-size: 0.78rem; color: var(--muted); font-weight: 600; margin-bottom: 4px; }
.conf-bar-bg { background: var(--sand); border-radius: 50px; height: 8px; }
.conf-bar-fill {
    height: 8px;
    border-radius: 50px;
    background: linear-gradient(90deg, #FF6B2B, #F4A22D);
    transition: width 0.8s ease;
}

/* ── HISTORY ── */
.history-item {
    background: var(--sand);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    cursor: pointer;
    border: 1px solid var(--border);
}
.history-item:hover { border-color: var(--saffron); }

/* ── DIVIDER ── */
.section-divider {
    border: none;
    border-top: 2px dashed var(--border);
    margin: 1.2rem 0;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 2px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FF6B2B, #F4A22D) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.2rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--saffron) !important; }

/* ── Audio uploader ── */
.stFileUploader > div {
    border-radius: 14px !important;
    border: 2px dashed var(--border) !important;
    background: var(--sand) !important;
}

/* ── Success/Warning boxes ── */
.stSuccess { background: #F0FFF4 !important; border-left-color: var(--success) !important; border-radius: 12px !important; }
.stWarning { background: #FFFBEB !important; border-radius: 12px !important; }
.stInfo    { background: #EEF2FF !important; border-radius: 12px !important; }

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TRAINING DATA
# ──────────────────────────────────────────────
@st.cache_resource
def train_model():
    data = {
        "text": [
            # SCHOLARSHIP (80)
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

            # AGRICULTURE (80)
            "farmer loan scheme","crop subsidy","agriculture support scheme","kisan yojana",
            "farmer needs money","crop damage help","subsidy for seeds","kheti ke liye paise chahiye",
            "fasal kharab ho gayi","kisan loan chahiye","beej ke liye paise chahiye",
            "tractor ke liye loan chahiye","kisan ke liye yojana","kheti ke liye paisa chahiye",
            "fasal kharab ho gayi","kisanon ke liye sahayata","kheti me nuksaan ho gaya",
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

            # EMPLOYMENT (80)
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

            # WOMEN SUPPORT (80)
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
            "mahila ke liye free education scheme kya hai","working women ke liye govt support kya hai",
            "mahila self employment ke liye kya help milti hai","ladkiyon ke liye safety apps ya schemes kya hai",
            "women ke liye financial independence kaise milegi","help for women",

            # HEALTH (80)
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
            "govt health support kaise le"," free health checkup kaha available hai",
            "govt hospital me treatment kaise start kare"
        ],

        "intent": (
            ["scholarship"] * 80 +
            ["agriculture"] * 80 +
            ["employment"] * 80 +
            ["women_support"] * 80 +
            ["health"] * 80
        )
    }

    df = pd.DataFrame(data)

    def clean_text(text):
        text = text.lower()
        cleaned = ""
        for char in text:
            if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
                cleaned += char
        return cleaned

    df["clean_text"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["intent"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return vectorizer, model, acc


# ──────────────────────────────────────────────
# SCHEME DATABASE
# ──────────────────────────────────────────────
GT = "https://translate.google.com/translate?sl=en&tl=hi&u="

schemes = {
    "scholarship": {
        "label": "📚 शिक्षा / छात्रवृत्ति",
        "color": "#4338CA",
        "schemes": [
            ("📚 नेशनल स्कॉलरशिप पोर्टल (NSP)",
             "✅ पात्रता: कक्षा 1 से PG | परिवार की आय ₹2.5 लाख से कम",
             GT + "scholarships.gov.in"),
            ("📚 पोस्ट मैट्रिक छात्रवृत्ति",
             "✅ पात्रता: SC/ST/OBC छात्र | कक्षा 11 के बाद",
             GT + "socialjustice.gov.in"),
            ("📚 PM यशस्वी योजना — ₹75,000–₹1,25,000",
             "✅ पात्रता: OBC/EBC/DNT छात्र | कक्षा 9 या 11",
             GT + "yet.nta.ac.in"),
            ("📚 विद्या लक्ष्मी — एजुकेशन लोन",
             "✅ पात्रता: कोई भी छात्र जिसे लोन चाहिए",
             GT + "vidyalakshmi.co.in"),
            ("📚 केंद्रीय क्षेत्र छात्रवृत्ति",
             "✅ पात्रता: 12वीं में 80%+ | परिवार की आय ₹4.5 लाख से कम",
             GT + "scholarships.gov.in"),
        ]
    },
    "agriculture": {
        "label": "🌾 खेती / किसान",
        "color": "#16A34A",
        "schemes": [
            ("🌾 PM किसान सम्मान निधि — हर साल ₹6,000",
             "✅ पात्रता: सभी किसान जिनके पास खेती की ज़मीन है",
             GT + "pmkisan.gov.in"),
            ("🌾 PM फसल बीमा योजना",
             "✅ पात्रता: सभी किसान | फसल खराब होने पर मुआवजा",
             GT + "pmfby.gov.in"),
            ("🌾 किसान क्रेडिट कार्ड (KCC)",
             "✅ पात्रता: सभी किसान | 4% ब्याज दर पर लोन",
             GT + "pmkisan.gov.in"),
            ("🌾 PM कृषि सिंचाई योजना",
             "✅ पात्रता: सभी किसान | सिंचाई उपकरण पर सब्सिडी",
             GT + "pmksy.gov.in"),
            ("🌾 सॉइल हेल्थ कार्ड — मुफ्त मिट्टी जाँच",
             "✅ पात्रता: सभी किसान | हर 2 साल में मुफ्त",
             GT + "soilhealth.dac.gov.in"),
        ]
    },
    "employment": {
        "label": "💼 रोजगार / नौकरी",
        "color": "#D97706",
        "schemes": [
            ("💼 मनरेगा — 100 दिन काम की गारंटी",
             "✅ पात्रता: ग्रामीण क्षेत्र के वयस्क | जॉब कार्ड ज़रूरी",
             GT + "nrega.nic.in"),
            ("💼 PM कौशल विकास योजना — मुफ्त ट्रेनिंग",
             "✅ पात्रता: 15–45 साल के युवा | 10वीं पास या ड्रॉपआउट",
             GT + "pmkvyofficial.org"),
            ("💼 DDU-GKY — ग्रामीण युवाओं के लिए",
             "✅ पात्रता: 15–35 साल | BPL परिवार को प्राथमिकता",
             GT + "ddugky.gov.in"),
            ("💼 PM रोजगार सृजन कार्यक्रम (PMEGP)",
             "✅ पात्रता: 18+ साल | खुद का व्यवसाय के लिए ₹25 लाख तक",
             GT + "kviconline.gov.in"),
            ("💼 नेशनल करियर सर्विस पोर्टल",
             "✅ पात्रता: सभी नौकरी ढूंढने वाले | मुफ्त रजिस्ट्रेशन",
             GT + "ncs.gov.in"),
        ]
    },
    "women_support": {
        "label": "👩 महिला सशक्तिकरण",
        "color": "#DB2777",
        "schemes": [
            ("👩 बेटी बचाओ बेटी पढ़ाओ",
             "✅ पात्रता: 0–10 साल की बेटियाँ | सभी परिवार",
             GT + "wcd.nic.in"),
            ("👩 PM उज्ज्वला योजना — मुफ्त गैस कनेक्शन",
             "✅ पात्रता: BPL परिवार की महिलाएं | APL भी पात्र",
             GT + "pmuy.gov.in"),
            ("👩 PM मातृत्व वंदना योजना — ₹5,000",
             "✅ पात्रता: गर्भवती व स्तनपान कराने वाली महिलाएं",
             GT + "pmmvy.wcd.gov.in"),
            ("👩 महिला शक्ति केंद्र",
             "✅ पात्रता: सभी ग्रामीण महिलाएं | कौशल व रोजगार सहायता",
             GT + "wcd.nic.in"),
            ("👩 स्वयं सहायता समूह (SHG) ऋण",
             "✅ पात्रता: महिलाएं | SHG के माध्यम से सस्ती ब्याज दर पर ऋण",
             GT + "nrlm.gov.in"),
        ]
    },
    "health": {
        "label": "🏥 स्वास्थ्य / इलाज",
        "color": "#DC2626",
        "schemes": [
            ("🏥 आयुष्मान भारत — ₹5 लाख तक मुफ्त इलाज",
             "✅ पात्रता: SECC 2011 सूची में नाम | BPL परिवार",
             GT + "pmjay.gov.in"),
            ("🏥 आयुष्मान कार्ड — यहाँ बनवाएं",
             "✅ अपना नाम चेक करें और कार्ड बनवाएं",
             GT + "beneficiary.nha.gov.in"),
            ("🏥 राष्ट्रीय स्वास्थ्य बीमा योजना (RSBY)",
             "✅ पात्रता: BPL परिवार | ₹30,000 तक बीमा",
             GT + "rsby.gov.in"),
            ("🏥 जननी सुरक्षा योजना",
             "✅ पात्रता: गर्भवती महिलाएं | BPL / SC / ST",
             GT + "nhm.gov.in"),
            ("🏥 Helpline: Ayushman 📞 14555 | NHM 📞 1800-180-1104",
             "✅ किसी भी सवाल के लिए — बिल्कुल मुफ्त",
             None),
        ]
    }
}


# ──────────────────────────────────────────────
# CHATBOT LOGIC
# ──────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    cleaned = ""
    for char in text:
        if char.isalnum() or char == " " or ("\u0900" <= char <= "\u097F"):
            cleaned += char
    return cleaned


def chatbot_response(user_input, vectorizer, model):
    if not user_input.strip():
        return None, None, 0

    # Translate if English
    try:
        lang = detect(user_input)
        if lang == "en":
            translated = GoogleTranslator(source="en", target="hi").translate(user_input)
        else:
            translated = user_input
    except Exception:
        translated = user_input

    cleaned = clean_text(translated)
    vec = vectorizer.transform([cleaned])
    intent = model.predict(vec)[0]
    confidence = float(model.predict_proba(vec).max()) * 100

    return intent, schemes[intent], confidence


def speech_to_text_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="hi-IN")
        return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        return None


# ──────────────────────────────────────────────
# INIT SESSION STATE
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""

vectorizer, model, model_acc = train_model()

# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🇮🇳 सरकारी <span>योजना</span> सहायक</div>
    <div class="hero-sub">ग्रामीण नागरिकों के लिए — हिंदी, हिंग्लिश या अंग्रेजी में पूछें</div>
    <div class="hero-badges">
        <span class="badge">📚 शिक्षा</span>
        <span class="badge">🌾 कृषि</span>
        <span class="badge">💼 रोजगार</span>
        <span class="badge">👩 महिला</span>
        <span class="badge">🏥 स्वास्थ्य</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LAYOUT: Two Columns
# ──────────────────────────────────────────────
col_main, col_side = st.columns([2.6, 1], gap="large")

with col_main:
    # ── TABS ──
    tab1, tab2 = st.tabs(["⌨️  टेक्स्ट से पूछें", "🎤  आवाज़ से पूछें"])

    with tab1:
        st.markdown('<div class="input-label">अपनी ज़रूरत लिखें</div>', unsafe_allow_html=True)
        user_input = st.text_input(
            label="query",
            label_visibility="collapsed",
            placeholder="जैसे: मुझे पढ़ाई के लिए पैसे चाहिए  |  kisan loan chahiye  |  I need a job",
            value=st.session_state.query,
            key="text_input"
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            search_btn = st.button("🔍 योजना खोजें", type="primary", use_container_width=True)
        with c2:
            clear_btn = st.button("🗑 साफ करें", type="secondary")

        if clear_btn:
            st.session_state.query = ""
            st.rerun()

        if search_btn and user_input.strip():
            with st.spinner("योजनाएं ढूंढ रहे हैं..."):
                intent, scheme_data, confidence = chatbot_response(user_input, vectorizer, model)

            if intent:
                st.session_state.history.insert(0, {"query": user_input, "intent": intent})
                if len(st.session_state.history) > 8:
                    st.session_state.history = st.session_state.history[:8]

                # Result section
                st.markdown(f"""
                <div class="result-section">
                    <div class="result-heading">
                        ✅ आपकी जरूरत: {scheme_data['label']}
                    </div>
                """, unsafe_allow_html=True)

                for name, elig, link in scheme_data["schemes"]:
                    link_html = f'<div class="scheme-link"><a href="{link}" target="_blank">🔗 हिंदी में वेबसाइट खोलें</a></div>' if link else ""
                    st.markdown(f"""
                    <div class="scheme-card">
                        <div class="scheme-name">{name}</div>
                        <div class="scheme-eligibility">{elig}</div>
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence bar
                bar_width = int(confidence)
                st.markdown(f"""
                <div class="confidence-wrap">
                    <div class="conf-label">AI विश्वास स्तर — {bar_width:.0f}%</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Helpline footer
                st.markdown("""
                <div style="background:#F0FDF4; border-radius:12px; padding:0.8rem 1.2rem; margin-top:0.8rem; border:1px solid #BBF7D0;">
                    💡 <b>ज़्यादा जानकारी के लिए:</b> नज़दीकी CSC या ग्राम पंचायत जाएं<br>
                    📞 <b>PM Kisan:</b> 1800-11-0001 &nbsp;&nbsp; 📞 <b>Ayushman:</b> 14555
                </div>
                """, unsafe_allow_html=True)

        elif search_btn and not user_input.strip():
            st.warning("⚠️ कृपया पहले अपनी ज़रूरत लिखें।")

    with tab2:
        st.info("🎤 अपनी आवाज़ में पूछें — एक WAV फ़ाइल अपलोड करें (हिंदी में बोलें)")
        st.markdown("🎤 बटन दबाएं और बोलें")

voice_text = speech_to_text(
    language='hi-IN',
    start_prompt="🎙️ बोलें",
    stop_prompt="⏹️ रोकें",
    just_once=True,
    use_container_width=True,
    key="mic"
)

if voice_text:
    st.success(f"🎤 आपने बोला: {voice_text}")

    intent, scheme_data, confidence = chatbot_response(
        voice_text, vectorizer, model
    )

    if intent:
        st.markdown(f"**✅ आपकी जरूरत:** {scheme_data['label']}")
        for name, elig, link in scheme_data["schemes"]:
            link_html = f' — [🔗 वेबसाइट]({link})' if link else ""
            st.markdown(f"- **{name}** | {elig}{link_html}")

        if audio_file:
            with st.spinner("आवाज़ पहचान रहे हैं..."):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                recognized = speech_to_text_from_file(tmp_path)
                os.unlink(tmp_path)

            if recognized:
                st.success(f"🎤 आपने बोला: **{recognized}**")
                intent, scheme_data, confidence = chatbot_response(recognized, vectorizer, model)
                if intent:
                    st.markdown(f"**✅ आपकी जरूरत:** {scheme_data['label']}")
                    for name, elig, link in scheme_data["schemes"]:
                        link_html = f' — [🔗 वेबसाइट]({link})' if link else ""
                        st.markdown(f"- **{name}** | {elig}{link_html}")
            else:
                st.error("⚠️ आवाज़ समझ नहीं आई। कृपया साफ़ हिंदी में बोलें और फिर से अपलोड करें।")

    # ── EXAMPLE QUESTIONS ──
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="examples-section">
        <div class="examples-title">📝 उदाहरण सवाल — क्लिक करके देखें</div>
        <div>
    """, unsafe_allow_html=True)

    examples = [
        "मुझे पढ़ाई के लिए पैसे चाहिए",
        "kisan loan chahiye fasal kharab ho gayi",
        "naukri nahi mil rahi help chahiye",
        "mahila ke liye koi yojana hai kya",
        "hospital ka kharcha bahut zyada hai",
        "mere pass apni beti ko padhane ke paise nahi hai",
        "I need a job urgently",
        "free medical treatment kaha milega",
    ]

    ex_cols = st.columns(4)
    for i, ex in enumerate(examples):
        with ex_cols[i % 4]:
            if st.button(ex, key=f"ex_{i}", help=ex, use_container_width=True):
                st.session_state.query = ex
                st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR COLUMN
# ──────────────────────────────────────────────
with col_side:

    # Model info card
    st.markdown(f"""
    <div style="background:white; border-radius:16px; padding:1.2rem 1.4rem; border:2px solid #E8D5B7; margin-bottom:1.2rem;">
        <div style="font-size:0.75rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#6B7280; margin-bottom:0.8rem;">
            🤖 मॉडल जानकारी
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.88rem;">
            <span style="color:#6B7280;">सटीकता</span>
            <span style="font-weight:700; color:#1B4332;">{model_acc*100:.1f}%</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-size:0.88rem;">
            <span style="color:#6B7280;">श्रेणियाँ</span>
            <span style="font-weight:700; color:#1B4332;">5</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.88rem;">
            <span style="color:#6B7280;">भाषाएं</span>
            <span style="font-weight:700; color:#1B4332;">हिंदी, हिंग्लिश, English</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Category guide
    cat_info = [
        ("📚", "शिक्षा", "Scholarship, fees, study"),
        ("🌾", "कृषि", "Kisan, farming, loan"),
        ("💼", "रोजगार", "Job, training, naukri"),
        ("👩", "महिला", "Women, beti, mahila"),
        ("🏥", "स्वास्थ्य", "Hospital, ilaj, treatment"),
    ]
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:1.2rem 1.4rem; border:2px solid #E8D5B7; margin-bottom:1.2rem;">
        <div style="font-size:0.75rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#6B7280; margin-bottom:0.8rem;">
            📂 श्रेणियाँ
        </div>
    """, unsafe_allow_html=True)
    for icon, name, hint in cat_info:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding:7px 0; border-bottom:1px dashed #F5E6CC;">
            <span style="font-size:1.2rem;">{icon}</span>
            <div>
                <div style="font-weight:600; font-size:0.9rem;">{name}</div>
                <div style="font-size:0.75rem; color:#9CA3AF;">{hint}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recent history
    if st.session_state.history:
        st.markdown("""
        <div style="background:white; border-radius:16px; padding:1.2rem 1.4rem; border:2px solid #E8D5B7;">
            <div style="font-size:0.75rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#6B7280; margin-bottom:0.8rem;">
                🕐 हाल की खोजें
            </div>
        """, unsafe_allow_html=True)
        for item in st.session_state.history[:5]:
            label = schemes[item["intent"]]["label"]
            st.markdown(f"""
            <div class="history-item">
                <div style="font-size:0.82rem; color:#374151; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{item['query'][:40]}...</div>
                <div style="font-size:0.72rem; color:#9CA3AF; margin-top:2px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑 इतिहास साफ करें", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer-strip">
    <span>🇮🇳 सरकारी योजना सहायक — ग्रामीण भारत के लिए</span>
    <span>📞 <a href="tel:18001110001">PM Kisan: 1800-11-0001</a> &nbsp;|&nbsp; <a href="tel:14555">Ayushman: 14555</a></span>
    <span style="opacity:0.7; font-size:0.78rem;">Built with ❤️ for Rural India</span>
</div>
""", unsafe_allow_html=True)
