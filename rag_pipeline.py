import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

warnings.filterwarnings("ignore")

print("1. Initializing RAG Knowledge Base (Organizational Security Policies)...")
# سیاست‌های رسمی سازمان که در پایگاه دانش برداری (Vector Store) ذخیره می‌شوند
trusted_policies = [
    "IT and Support will never send a link requiring password retention or account verification via external portals.",
    "Webmail account updates and password maintenance are handled automatically without user manual validation.",
    "All proforma invoice requests with compressed attachments (like .7z or .shtml) must be flagged for security inspection.",
    "Official notices regarding legal or administrative procedures are sent through verified government or official domain channels only.",
    "Commercial software promotions and marketing emails for IMED, ISO 13485, or TTAC XML label generators are unauthorized external promotions."
]

# ساخت پایگاه دانش برداری با مدل امبدینگ سبک و پرسرعت
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(trusted_policies, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

print("2. Loading Content Understanding NLP Classifier...")
# مدل طبقه‌بندی معنایی برای تشخیص نیت پیام
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
candidate_labels = [
    "password expiration urgency",
    "account verification phish",
    "unauthorized commercial promotion",
    "suspicious order or invoice request",
    "normal internal task communication"
]

print("3. Analyzing All Email Cases from the Word File...\n")

# تمام متون فایل ورد ارسالی شما به تفکیک
emails_from_doc = [
    {
        "id": "Email 1",
        "sender": "vutha@vimpex.hu (Claiming COMPANY.com)",
        "body": "USER@COMPANY.com رمز عبور شما ظرف ۲۴ ساعت منقضی می‌شود. برای ادامه با همان رمز عبور و جلوگیری از قطع شدن اتصال، از دکمه زیر استفاده کنید. همان رمز عبور را نگه دارید."
    },
    {
        "id": "Email 2",
        "sender": "vutha@vimpex.hu",
        "body": "برخی از پیام های شما مسدود شد. لطفاً ایمیل خود را در زیر تأیید کنید تا همه پیام‌های مسدود شده را به صندوق ورودی خود بازگردانید. در اینجا تأیید کنید."
    },
    {
        "id": "Email 3",
        "sender": "COMPANY.com Server Notice",
        "body": "رمز عبور صندوق پست شما امروز منقضی می‌شود. پس از این پیام، وب‌میل شما شما را خارج کرده و یک رمز عبور جدید ایجاد خواهد کرد. ما پیشنهاد می‌کنیم رمز عبور فعلی خود را نگه دارید."
    },
    {
        "id": "Email 4",
        "sender": "سیب سبز سلامت (Www.ghapple.com)",
        "body": "فرایند اخذ IMED تولید: ۱- نگارش تکنیکال فایل ۲- پیاده سازی ایزو ۱۳۴۸۵ ۳- آماده سازی خط تولید ۴- گزارش آزمون محصولات ۵- بازرسی حضوری و صدور پروانه ساخت جهت مشاوره رایگان با ما در تماس باشید."
    },
    {
        "id": "Email 5",
        "sender": "پیشرو سلامت پارس",
        "body": "نرم افزار تولید برچسب اصالت کالا، یک نرم افزار تحت ویندوز است که توسط آن تولیدکنندگان و واردکنندگان اقلام سلامت محور می توانند فایل ایکس ام ال اصالت کالا برای سامانه تیتک تولید کنند."
    },
    {
        "id": "Email 6",
        "sender": "Unknown Commercial",
        "body": "روز بخیر، لطفاً بهترین پیشنهاد قیمتی خود را برای اقلام مشابهِ پیوست‌شده اعلام فرمایید. تاریخ تخمینی ورود (ETA) به بندر عباس. Attachment: 130 MT Acetic Acid_PDF.shtml"
    },
    {
        "id": "Email 7",
        "sender": "Sales@ceramtec.com.cn",
        "body": "Notice of Webmail Deactivation Policy (2025/2026). اطلاعیه به‌روزرسانی وب‌میل. ما در حال حاضر در حال به‌روزرسانی تمام حساب‌های وب‌میل فعال به نسخه‌ای جدیدتر هستیم. برای شروع به پورتال تأیید مراجعه کنید."
    },
    {
        "id": "Email 8",
        "sender": "info@unismack.gr",
        "body": "Hello Sir, Attached is the list of our new order. Please send me a proforma invoice so that I can make the payment. Attachment: TELEX HBL 42768397623819-HBL.7z"
    },
    {
        "id": "Email 9",
        "sender": "Internal Staff",
        "body": "با سلام. مدیر محترم فروش، لطفا از ردیف 1602 به بعد(ردیف آبی رنگ) زحمت بکشید و بر روی دیتای سیستم ویرایش انجام دهید. با احترام"
    }
]

# اجرای خط لوله تشخیص RAG روی تک‌تک داده‌ها
for item in emails_from_doc:
    print(f"=== [Analyzing {item['id']}] Sender: {item['sender']} ===")
    
    # لایه درک معنایی (Intent Detection)
    intent_result = classifier(item['body'], candidate_labels)
    top_intent = intent_result['labels'][0]
    score = intent_result['scores'][0]
    print(f"[*] Detected Intent: {top_intent} (Confidence: {score:.2f})")
    
    # لایه اعتبارسنجی RAG
    retrieved_docs = retriever.invoke(item['body'])
    matched_policy = retrieved_docs[0].page_content
    print(f"[*] RAG Policy Reference: {matched_policy}")
    
    # تولید هشدار هوشمند و تصمیم‌گیری
    if "urgency" in top_intent or "phish" in top_intent or "invoice" in top_intent:
        print("[!] DECISION: HIGH RISK - QUARANTINED")
        print(f"[!] EXPLANATION: Threat detected ({top_intent}) which violates corporate policy.\n")
    elif "commercial" in top_intent:
        print("[!] DECISION: MEDIUM RISK - SPAM / PROMOTION")
        print(f"[!] EXPLANATION: External marketing content detected.\n")
    else:
        print("[+] DECISION: SAFE - PASSED\n")