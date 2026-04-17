python -m venv venv

venv\Scripts\activate

pip install -r rtc/requirements.txt

cd rtc

uvicorn main:app --reload

streamlit run app.py
