# 🚀 دليل تشغيل Retail Analytics Copilot عبر Google Colab + Ngrok

## نظرة عامة على الـ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                LOCAL_MODELS=False                       │
│                                                         │
│  ┌──────────┐    ┌──────────────────────────────────┐  │
│  │  UI /    │    │          Google Colab             │  │
│  │  FastAPI │◄──►│  Ollama + Ngrok Tunnel            │  │
│  │  Backend │    │  (notebook لكل موديل)             │  │
│  └──────────┘    └──────────────────────────────────┘  │
│                                                         │
│  الـ UI يعمل محلياً ويتصل بالموديلات عبر Ngrok          │
└─────────────────────────────────────────────────────────┘
```

## الموديلات والـ Notebooks

| Notebook | الموديل | الدور | متغير .env |
|---|---|---|---|
| `router_model_colab.ipynb` | `phi3.5:3.8b-mini-instruct-q4_K_M` | تصنيف السؤال: rag/sql/hybrid | `NGROK_ROUTER_URL` |
| `nl2sql_model_colab.ipynb` | `gemma2:9b-instruct-q5_0` | تحويل السؤال إلى SQL | `NGROK_NL2SQL_URL` |
| `synthesis_model_colab.ipynb` | `gemma2:9b-instruct-q5_0` | توليد الإجابة النهائية | `NGROK_SYNTHESIS_URL` |
| `planner_model_colab.ipynb` | `phi3.5:3.8b-mini-instruct-q4_K_M` | استخراج القيود والتواريخ | `NGROK_PLANNER_URL` |

---

## خطوات التشغيل

### الخطوة 1: إنشاء حساب Ngrok
1. اذهب إلى https://ngrok.com وأنشئ حساباً مجانياً
2. في Dashboard، انسخ **Auth Token** من صفحة `Your Authtoken`

### الخطوة 2: إضافة Ngrok Token إلى Colab Secrets
في كل notebook:
1. افتح **Colab Secrets** (أيقونة 🔑 في الشريط الجانبي الأيسر)
2. اضغط `+ Add new secret`
3. الاسم: `NGROK_AUTH_TOKEN`
4. القيمة: الـ token الذي نسخته من Ngrok

### الخطوة 3: تشغيل الـ Notebooks
افتح **كل notebook على Colab منفصل** وشغّل جميع الخلايا بالترتيب:

```
router_model_colab.ipynb    → ستحصل على NGROK_ROUTER_URL
nl2sql_model_colab.ipynb    → ستحصل على NGROK_NL2SQL_URL
synthesis_model_colab.ipynb → ستحصل على NGROK_SYNTHESIS_URL
planner_model_colab.ipynb   → ستحصل على NGROK_PLANNER_URL
```

> ⚠️ **مهم:** تأكد من تفعيل GPU في كل Colab:
> Runtime > Change runtime type > T4 GPU

### الخطوة 4: تحديث ملف .env

```env
LOCAL_MODELS=False

NGROK_ROUTER_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M
NGROK_ROUTER_URL=https://xxxx-xxx.ngrok-free.app

NGROK_NL2SQL_MODEL=gemma2:9b-instruct-q5_0
NGROK_NL2SQL_URL=https://yyyy-yyy.ngrok-free.app

NGROK_SYNTHESIS_MODEL=gemma2:9b-instruct-q5_0
NGROK_SYNTHESIS_URL=https://zzzz-zzz.ngrok-free.app

NGROK_PLANNER_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M
NGROK_PLANNER_URL=https://wwww-www.ngrok-free.app
```

### الخطوة 5: تشغيل الـ Backend محلياً

```bash
# تأكد من تثبيت المتطلبات
pip install -r requirements.txt

# شغّل الـ API Server (يتصل بالموديلات عبر Ngrok تلقائياً)
python api_server.py
```

### الخطوة 6: فتح الـ UI
افتح المتصفح على: `http://localhost:8000`

---

## ملاحظات مهمة

### حول Ngrok المجاني
- الحساب المجاني يسمح بـ **tunnel واحد** في وقت واحد فقط
- إذا أردت 4 tunnels في نفس الوقت، تحتاج حساب **Ngrok Pro**
- **البديل:** شغّل كل notebook واحدة تلو الأخرى وغيّر الـ URL في .env

### حول Google Colab
- Colab يوقف الـ session بعد ~90 دقيقة من عدم التفاعل
- بعد انتهاء الـ session، ستتغير روابط Ngrok وتحتاج لتحديث .env
- استخدم `Colab Pro` للحصول على sessions أطول

### حول الـ fallback
- إذا كان أي `NGROK_*_URL` غير محدد، يستخدم الكود `DummyLM` تلقائياً
- لا يوجد crash في الـ app — فقط إجابات تجريبية حتى تضبط الروابط

---

## الفرق بين LOCAL_MODELS=True و False

| | `LOCAL_MODELS=True` | `LOCAL_MODELS=False` |
|---|---|---|
| الموديلات | Ollama محلياً | Google Colab + Ngrok |
| الـ GPU | GPU/CPU المحلي | T4 GPU مجاني من Colab |
| الإعداد | بسيط — Ollama فقط | يحتاج 4 Colab sessions |
| الأداء | حسب جهازك | T4 GPU سريع |
| الـ Latency | منخفض | أعلى (internet round-trip) |
| الحاجة للإنترنت | لا (بعد تحميل الموديل) | نعم دائماً |

