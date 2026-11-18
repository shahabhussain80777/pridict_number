# RNN vs LSTM vs GRU – Next Value Predictor

سادہ ٹائم سیریز (روزانہ درجہ حرارت، 500 دن) پر RNN, LSTM اور GRU ٹرین کیے گئے۔
3 ویلیو دینے پر تینوں ماڈل اگلی ویلیو پریڈکٹ کرتے ہیں۔ Vercel پر ڈپلائے ہوتا ہے۔

## Structure
```
.
├── api/
│   ├── predict.py            # Vercel serverless function (NumPy inference)
│   └── model_weights.json    # ٹرین شدہ ویٹس + RMSE
├── public/
│   └── index.html            # 3 ویلیو والا ویب پیج
├── training/
│   ├── train.py              # پورا ٹریننگ کوڈ (load, windows, train, graph, RMSE)
│   ├── rnn_models.py         # RNN / LSTM / GRU (NumPy)
│   ├── data.csv              # ڈیٹا سیٹ
│   ├── results.png           # گراف: Actual vs RNN vs LSTM vs GRU
│   ├── rmse_results.txt
│   └── requirements.txt      # ٹریننگ کے لیے
├── requirements.txt          # Vercel کے لیے (صرف numpy)
├── vercel.json
└── README.md
```

## دوبارہ ٹرین کرنا (اختیاری)
```
cd training
pip install -r requirements.txt
python train.py
```
یہ `api/model_weights.json` دوبارہ بنا دیتا ہے۔

## Vercel پر ڈپلائے
1. پورا فولڈر GitHub repo میں push کریں (root پر `api/`, `public/`, `requirements.txt`, `vercel.json` ہوں)۔
2. vercel.com → Add New Project → وہ repo منتخب کریں → Deploy (کوئی سیٹنگ بدلنے کی ضرورت نہیں، Framework: Other)۔

## Results (test data, last 20%)
| Model | RMSE (°C) |
|-------|-----------|
| RNN   | 0.8457 |
| LSTM  | 0.8094 |
| GRU   | 0.8098 |

### Final Answer
1. Lowest RMSE: **LSTM** (0.8094)
2. Best prediction: **LSTM**, GRU تقریباً برابر (فرق صرف 0.0003)
3. RNN کے مقابلے میں LSTM اور GRU دونوں بہتر رہے؛ RNN سب سے کمزور (0.8457)۔
