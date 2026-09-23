# Credit Score Classification - Streamlit Deployment

## 1. Train and export
Open `Capstone_credit_score_15_2209_V2_DEPLOYABLE.ipynb` in the same folder as `credit_score_preprocessor.py`.

Run the notebook through the final model comparison and then run the final **DEPLOYMENT EXPORT** cell.

That cell creates:
`models_export/credit_score_pipeline.joblib`

## 2. Test locally
Install:
`pip install -r requirements.txt`

Then:
`streamlit run streamlit_app.py`

## 3. GitHub structure
credit-score-classification/
- streamlit_app.py
- requirements.txt
- credit_score_preprocessor.py
- models_export/
  - credit_score_pipeline.joblib

The notebook does not need to be deployed for the app to run.
