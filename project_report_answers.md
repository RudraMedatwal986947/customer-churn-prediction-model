### b) Brief Overview of Work Completed After Reporting 1
Following the initial setup and database design, development transitioned into implementing the core Machine Learning, API, and Frontend components of the platform:
* **Machine Learning Pipeline Engineering**: Developed a robust data preprocessing pipeline that performs automated missing value imputation, engineered key features (like `tenure_group` and `avg_monthly_charge`), and applies One-Hot Encoding and Scaling.
* **Model Training & Serialization**: Implemented K-Means clustering for customer segmentation. Trained high-performing XGBoost models—a Classifier for churn risk prediction and a Regressor for Customer Lifetime Value (CLV) estimation—and serialized them for production use.
* **Backend API Development**: Built a FastAPI REST backend capable of dynamically loading the serialized ML models to serve real-time inferences. Created endpoints to serve analytical insights directly from the PostgreSQL database.
* **Frontend Dashboard Integration**: Built a fully interactive Streamlit web dashboard utilizing Plotly for Exploratory Data Analysis (EDA) and cluster visualizations. Wired the dashboard directly to the FastAPI endpoints to create a seamless user interface for running live predictions.
* **Quality Assurance**: Designed and implemented mocked unit test suites utilizing `pytest` to guarantee the reliability of both the data preprocessing logic and the backend API routes.

***

### c) System Architecture / Workflow Diagram
The platform is orchestrated via Docker and divided into three primary interconnected layers (Data, Logic/API, and Presentation). Below is the architecture flowchart diagram:

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
    classDef backend fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white;
    classDef database fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:black;
    classDef ml fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:white;

    %% Components
    User((End User / Stakeholder))
    
    subgraph Presentation Layer
        UI[Streamlit Dashboard Interface]:::frontend
        UI_EDA[EDA & Visualizations]:::frontend
        UI_Seg[Segmentation View]:::frontend
        UI_Pred[Live Predictions UI]:::frontend
    end
    
    subgraph Logic & API Layer
        API[FastAPI Backend Server]:::backend
        Router_Pred[/predict Router]:::backend
        Router_Ins[/insights Router]:::backend
    end
    
    subgraph Machine Learning Layer
        Preprocess[Data Preprocessor & Scaler]:::ml
        Model_Churn[XGBoost Churn Classifier]:::ml
        Model_CLV[XGBoost CLV Regressor]:::ml
        Model_Seg[K-Means Clustering]:::ml
    end
    
    subgraph Data Layer
        DB[(PostgreSQL Database)]:::database
    end

    %% Connections
    User -->|Interacts with| UI
    UI --- UI_EDA
    UI --- UI_Seg
    UI --- UI_Pred
    
    UI_Pred -->|HTTP POST Request| Router_Pred
    UI_Seg -->|HTTP GET Request| Router_Ins
    UI_EDA -.->|Direct SQL Query| DB
    
    API --- Router_Pred
    API --- Router_Ins
    
    Router_Pred -->|Loads & Calls| Model_Churn
    Router_Pred -->|Loads & Calls| Model_CLV
    Router_Ins -->|Fetches Stats| DB
    
    Model_Seg -->|Pushes Segments| DB
    Preprocess -->|Feeds Data| Model_Churn
    Preprocess -->|Feeds Data| Model_CLV
    
    DB -->|Historical Data| Preprocess
```

***

### d) Modules Completed / Features Implemented

| Module Name | Description | Current Status |
| :--- | :--- | :---: |
| **Database & Ingestion Module** | Configured `docker-compose.yml` for service orchestration. Built SQLAlchemy ORM models (`database/models.py`) mapping to PostgreSQL. Developed `database/seed_db.py` to parse and ingest the raw Telco Excel dataset into the database. | ✅ Completed |
| **Data Preprocessing Module** | `ml/data_preprocessing.py` — Robust pipeline handling missing value imputation, feature engineering (`tenure_group`, `avg_monthly_charge`), One-Hot Encoding, binary encoding, and Standard Scaling for both training and production inference. | ✅ Completed |
| **Customer Segmentation Module** | `ml/segmentation.py` — Extracts RFM (Recency, Frequency, Monetary) features and applies K-Means Clustering (4 clusters) to group customers into distinct behavioral cohorts. Pushes segment labels back to the database. | ✅ Completed |
| **Churn Prediction Module** | `ml/churn_prediction.py` — Trains an XGBoost Classifier on labelled churn data achieving ~80% accuracy. Serializes the trained model and scaler as `.pkl` files for API consumption. | ✅ Completed |
| **CLV Prediction Module** | `ml/clv_prediction.py` — Trains an XGBoost Regressor to estimate Customer Lifetime Value achieving an R² of 0.99. Serializes the model and its dedicated scaler for API consumption. | ✅ Completed |
| **Visualization Generation Module** | `ml/generate_plots.py` — Generates static PNG visualizations: ROC Curve, Confusion Matrix, Feature Importance plot, and Segmentation Scatter plot saved to the `visualizations/` directory. | ✅ Completed |
| **FastAPI Backend — Predict Routes** | `api/routes/predict.py` — Exposes `/api/v1/predict/churn` and `/api/v1/predict/clv` endpoints. Lazy-loads serialized `.pkl` models, queries the database for customer features, applies preprocessing, and returns JSON predictions with risk tiers. | ✅ Completed |
| **FastAPI Backend — Insights Routes** | `api/routes/insights.py` — Exposes `/api/v1/insights/segmentation/summary` and `/segmentation/customer` endpoints, returning SQL-aggregated segment statistics and per-customer cluster assignments. | ✅ Completed |
| **EDA Dashboard Page** | `dashboard/pages/1_eda.py` — Interactive Streamlit page fetching live data from PostgreSQL and rendering Plotly charts: Churn Distribution pie chart, Tenure histogram, Monthly Charges box plot, and Contract Type bar chart. | ✅ Completed |
| **Segmentation Dashboard Page** | `dashboard/pages/2_segmentation.py` — Streamlit page calling the Insights API to display cluster summary statistics, a Tenure vs. Monthly Charges scatter plot, and a Total Charges box plot segmented by cluster. | ✅ Completed |
| **Predictions Dashboard Page** | `dashboard/pages/3_predictions.py` — Interactive Streamlit inference UI. Allows selection of a Customer ID from the database, makes live HTTP calls to the FastAPI `/predict` endpoints, and renders color-coded churn risk level and CLV estimate. | ✅ Completed |
| **Automated Testing Module** | `tests/test_ml_preprocessing.py` — Pytest unit tests for the data pipeline using fixture DataFrames. `tests/test_api.py` — Mocked API route tests using `TestClient` and `unittest.mock` to validate JSON contracts without a live database. | ✅ Completed |
| **MLOps Retraining Pipeline** | Automated cron-based model retraining orchestrator to prevent model drift over time. | 🔜 Planned |
| **Authentication Module** | OAuth2 / JWT authentication layer for FastAPI prediction endpoints. | 🔜 Planned |
| **CI/CD Pipeline** | GitHub Actions workflow to auto-run tests, build Docker images, and deploy to cloud on every commit to `main`. | 🔜 Planned |

***

### e) Database
* **PostgreSQL (v15)**: Serves as the primary relational database for the platform. It securely stores raw customer profiles, billing data, service usage, and dynamically generated K-Means cluster assignments.
* **SQLAlchemy & Psycopg2**: Python Object-Relational Mapper (ORM) and PostgreSQL adapter used to programmatically interact with the database, allowing for seamless data ingestion and querying without writing raw SQL for every transaction.

### f) Hardware/Software Tools
* **Docker & Docker Compose**: Containerization platforms used to orchestrate the three-tier architecture (Database, API, Frontend) ensuring environmental consistency and seamless deployment across any hardware.
* **Python (3.9+)**: The core programming language running the entire stack.
* **Git**: Used for version control and repository management.

### g) Other Technologies
* **FastAPI & Uvicorn**: A modern, high-performance web framework for building asynchronous RESTful APIs, used to expose the Machine Learning models.
* **Streamlit**: An open-source app framework used to build the interactive, data-rich frontend dashboard entirely in Python.
* **XGBoost**: An optimized distributed gradient boosting library utilized for training the highly accurate Churn Classification and CLV Regression models.
* **Scikit-Learn**: Utilized for its K-Means clustering algorithm and robust data preprocessing utilities (StandardScaler, LabelEncoders).
* **Pandas & NumPy**: The foundational data manipulation and scientific computing libraries used heavily for feature engineering and dataset transformations.
* **Plotly**: A graphing library used within the Streamlit dashboard to render interactive, publication-quality visualizations.

***

### 4. Implementation Description

**Algorithm / Method followed**
The platform employs a multi-faceted machine learning approach. 
* For **Customer Segmentation**, the **K-Means Clustering** algorithm was used on RFM (Recency, Frequency, Monetary) engineered features to group customers into behavioral cohorts.
* For **Churn Prediction**, an **XGBoost Classifier** was trained to identify binary churn risk (Yes/No) based on demographic and service usage patterns.
* For **CLV Estimation**, an **XGBoost Regressor** was utilized to predict the continuous numeric value of a customer's lifetime revenue. 
All algorithms were fed data processed through One-Hot Encoding (for categorical variables) and Standard Scaling (for numerical stability).

**Coding / Development details**
Development was conducted entirely in Python, utilizing a modular, service-oriented architecture. The codebase is separated into distinct domains: `database/` for ORM models and seeding, `ml/` for data preprocessing and model training scripts, `api/` for the backend server, and `dashboard/` for the user interface. Models trained in the `ml` module were serialized using `joblib` so they could be decoupled and lazy-loaded dynamically by the API layer in production.

**Database design**
The system uses a relational PostgreSQL database. The core schema revolves around a `customers` table constructed via SQLAlchemy ORM. It stores static demographic info (gender, dependents), service configurations (internet type, security), and billing histories (monthly/total charges). The schema was designed to be extensible, allowing the ML pipelines to write back insights—dynamically populating `segment`, `predicted_churn`, and `predicted_clv` columns directly into the customer records for fast retrieval.

**API integration**
The backend was developed using FastAPI to serve RESTful endpoints. The `/api/v1/predict/` namespace accepts JSON payloads containing a `customer_id`. The API queries the database for that customer's raw features, runs them through the pre-loaded XGBoost scaler and model, and returns a JSON response containing the churn probability and risk tier. The Streamlit frontend acts as the client, utilizing the `requests` library to seamlessly hit these endpoints and render the JSON responses as visual metrics and gauges.

**Testing performed**
Testing was implemented using the `pytest` framework. 
* **Data Pipeline Testing**: Unit tests were written for the `preprocess_data` function using dummy pandas DataFrames (fixtures) to verify that features were engineered correctly, categorical columns were successfully dropped/encoded, and mathematical scaling was applied properly.
* **API Route Testing**: The FastAPI endpoints were tested utilizing `TestClient` combined with `unittest.mock`. Database calls (`pd.read_sql`) and model loading (`joblib.load`) were stubbed out, allowing the tests to validate that the API routing, exception handling (e.g., 404 for missing customers), and JSON response formatting worked reliably without needing a live database connection.

***

### 5. Results / Output Screenshots

**Experimental & Simulation Results**
The machine learning experiments yielded highly actionable business intelligence. The XGBoost models effectively identified at-risk customers, while the K-Means algorithm successfully mapped out distinct customer profiles based on spending and tenure.

**Screenshot 1: XGBoost Model Feature Importance**
*(Insert `visualizations/feature_importance.png` here)*
* **Description**: This plot illustrates the top drivers of customer churn identified by the XGBoost algorithm. Features such as contract type (month-to-month), tenure duration, and total monthly charges were identified as having the highest F-score importance in determining if a customer is likely to leave.

**Screenshot 2: XGBoost ROC Curve & Confusion Matrix**
*(Insert `visualizations/roc_curve.png` and `visualizations/confusion_matrix.png` here)*
* **Description**: The Receiver Operating Characteristic (ROC) curve validates the strong predictive performance and high Area Under the Curve (AUC) of the churn classifier. The confusion matrix highlights the model's accuracy in correctly flagging true positives (actual churners) against false positives.

**Screenshot 3: K-Means Customer Segmentation Scatter**
*(Insert `visualizations/segmentation_scatter.png` here)*
* **Description**: A visual scatter plot mapping the output of the K-Means clustering algorithm. It clearly demarcates distinct customer cohorts based on their recency, frequency, and monetary metrics, allowing marketing teams to target specific groups.

**Application Interface**

**Screenshot 4: Exploratory Data Analysis (EDA) Dashboard**
*(Insert a screenshot of `dashboard/pages/1_eda.py` running in your browser here)*
* **Description**: The interactive Streamlit frontend displaying live charts, giving stakeholders a high-level overview of the dataset's distribution, including churn rates, tenure length histograms, and contract-type breakdowns.

**Screenshot 5: Live Prediction Interface**
*(Insert a screenshot of `dashboard/pages/3_predictions.py` running in your browser here)*
* **Description**: The inference UI where a user inputs a `customer_id`. The dashboard makes a live REST API call to the FastAPI backend, dynamically returning the estimated Customer Lifetime Value and a color-coded Churn Risk probability gauge (e.g., High, Medium, Low).

***

### 6. Challenges Faced and Solutions

| Challenge Faced | Solution Applied |
| :--- | :--- |
| **Categorical Encoding Mismatch during Inference** <br> Performing One-Hot Encoding (`get_dummies`) on a single customer's data payload via the API resulted in a missing columns error, as the model expected the exact feature matrix shape generated during training on the full dataset. | During the API inference step, we loaded the historical dataset into context to ensure the `get_dummies` function maps the single user against all possible categories, successfully preserving the matrix dimensions expected by XGBoost. |
| **Frontend/Backend Coupling & Thread Blocking** <br> Running heavy machine learning inference directly inside the Streamlit frontend would block the main UI thread, causing sluggish user experiences and scaling bottlenecks. | Decoupled the architecture entirely. We wrapped the serialized `joblib` ML models inside an asynchronous FastAPI REST backend. Streamlit operates purely as a lightweight presentation layer making non-blocking HTTP requests. |
| **Environment & Dependency Hell** <br> Guaranteeing that the PostgreSQL database, FastAPI, and Streamlit apps communicate seamlessly with the exact Python versions and dependency libraries across different operating systems. | Containerized the entire platform using Docker. We authored a `docker-compose.yml` file to orchestrate the services with isolated internal networking, ensuring a single command (`docker-compose up`) spins up a reproducible environment anywhere. |
| **Hidden Missing Values in Raw Data** <br> The raw Telco dataset contained missing numerical values disguised as blank string spaces (e.g., in `total_charges`), causing cryptic `TypeError` exceptions during early model training. | Authored a strict data coercion step in `ml/data_preprocessing.py` using `pd.to_numeric(errors='coerce')`, aggressively exposing hidden strings as NaNs, followed by a logical zero-fill imputation strategy. |

***

### 7. Tasks in Progress / Next Month Plan

**Remaining modules**
While the core pipeline is fully implemented, the focus for the next iteration is adding production-grade features:
* **MLOps Retraining Pipeline**: Developing a cron-based orchestrator (e.g., using Apache Airflow) to automatically fetch new historical data from the PostgreSQL database and retrain the XGBoost models monthly to prevent model drift.
* **Authentication Module**: Implementing OAuth2 / JWT authentication within the FastAPI layer to secure the prediction endpoints against unauthorized access.

**Testing plan**
Our foundational unit testing via `pytest` is complete. The plan for next month is focused on robust quality assurance:
* **End-to-End (E2E) Testing**: Utilizing `docker-compose` to spin up ephemeral testing environments where the Streamlit frontend drives simulated user journeys to test the entire stack holistically.
* **Load Testing**: Utilizing tools like `Locust` to simulate high-concurrency traffic on the FastAPI `/predict` routes to identify latency bottlenecks before a full-scale launch.

**Documentation work**
* **API Documentation**: Finalizing the interactive OpenAPI (Swagger) documentation automatically generated by FastAPI, enriching the docstrings and adding example request payloads for third-party consumers.
* **Developer Onboarding**: Expanding the repository's `README.md` to include comprehensive architectural diagrams, local development setup guides, and environment variable documentation.

**Deployment plan**
* **Cloud Infrastructure Setup**: Provisioning managed instances on a cloud provider (e.g., AWS EC2 or Google Cloud Run) to host our three Dockerized services.
* **CI/CD Pipeline**: Establishing continuous integration and deployment pipelines using GitHub Actions. Upon a push to the `main` branch, the pipeline will automatically run our `pytest` suite, build the Docker images, and push them to a container registry before triggering a rolling update on the production servers.

***

### 8. Learning / Skills Acquired

**Technical Skills Learned:**
* **Predictive Modelling with XGBoost**: Gained hands-on experience training, evaluating, and serializing gradient-boosted classification and regression models for real-world business use cases (churn risk and revenue estimation).
* **Unsupervised Learning & Customer Segmentation**: Developed expertise in applying K-Means clustering combined with RFM (Recency, Frequency, Monetary) feature engineering to discover meaningful customer behavioral cohorts.
* **Data Engineering & Preprocessing Pipelines**: Built robust, reusable preprocessing pipelines in `scikit-learn` handling missing value imputation, One-Hot Encoding, binary encoding, and Standard Scaling for production-grade inference.
* **RESTful API Development with FastAPI**: Designed and implemented asynchronous REST endpoints following resource-oriented architecture for serving live ML model inferences and database aggregations.
* **Database Design & ORM with SQLAlchemy**: Designed and managed relational schemas in PostgreSQL using SQLAlchemy ORM, including reverse-writing ML prediction results back to the database.
* **Containerized System Architecture with Docker**: Architected and orchestrated a fully Dockerized three-tier system (Database → API → Frontend) using `docker-compose` for environment reproducibility.
* **Interactive Dashboard Development with Streamlit & Plotly**: Built rich, interactive data dashboards in Python using Streamlit with Plotly-powered visualizations (scatter plots, pie charts, histograms, box plots).
* **Automated Testing with Pytest & Mocking**: Wrote unit and integration tests using `pytest`, `TestClient`, and `unittest.mock` to validate ML pipeline integrity and API contract correctness without a live database.

**Professional Skills Learned:**
* **End-to-End System Thinking**: Developed the ability to visualize and implement the full lifecycle of a data product — from raw data ingestion through to user-facing dashboards.
* **Modular Code Design**: Practiced structuring large codebases into clearly separated, single-responsibility modules (`database/`, `ml/`, `api/`, `dashboard/`) for maintainability and scalability.
* **Technical Documentation**: Produced structured markdown documentation including implementation plans, project scopes, walkthroughs, and progress trackers to keep the project aligned and auditable.
* **Problem Decomposition & Debugging**: Gained experience diagnosing and resolving complex cross-layer issues, such as categorical encoding mismatches between training and inference contexts.
* **Agile & Iterative Development**: Adopted an iterative approach, building core infrastructure first, then incrementally layering ML models, APIs, and frontend components.

***

### 9. References

**Research Paper References:**

[1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System", *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.

[2] J. MacQueen, "Some Methods for Classification and Analysis of Multivariate Observations", *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1967.

[3] P. Óskarsdóttir, C. Bravo, W. Verbeke, C. Sarraute, B. Baesens and J. Vanthienen, "Social Network Analytics for Churn Prediction in Telco: Model Building, Explainability and Evaluation", *Expert Systems with Applications*, Elsevier, 2017.

**Website References:**

[4] FastAPI Documentation, "FastAPI — Modern, Fast Web Framework for Building APIs with Python", https://fastapi.tiangolo.com, Accessed July 2026.

[5] Streamlit Documentation, "Streamlit — A faster way to build and share data apps", https://docs.streamlit.io, Accessed July 2026.

[6] Scikit-Learn Documentation, "scikit-learn: Machine Learning in Python", https://scikit-learn.org/stable/documentation.html, Accessed July 2026.

[7] Docker Documentation, "Docker — Accelerated Container Application Development", https://docs.docker.com, Accessed July 2026.

[8] IBM, "Telco Customer Churn Dataset", IBM Sample Data Sets, https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets, Accessed July 2026.

**Book References:**

[9] A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Edition, O'Reilly Media, 2019.

[10] W. McKinney, *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython*, 2nd Edition, O'Reilly Media, 2018.
