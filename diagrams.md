# Reddit Sentiment Analysis System - Diagrams

## 1. Proposed System Diagram

```mermaid
graph TB
    subgraph "External Data Sources"
        Reddit[Reddit API]
        Gemini[Google Gemini API]
    end
    
    subgraph "Data Collection Layer"
        PRAW[PRAW Client]
        Fetch[fetch_data.py]
    end
    
    subgraph "Data Processing Layer"
        RawData[(Raw Data<br/>JSON)]
        Preprocess[Preprocessor<br/>clean_text]
        CleanedData[(Cleaned Data<br/>CSV)]
        VADER[VADER Analyzer]
        LabeledData[(Labeled Data<br/>CSV)]
    end
    
    subgraph "ML Training Layer"
        TFIDF[TF-IDF Vectorizer]
        Models[14 ML Models]
        Pipeline[Model Pipeline]
        TrainedModels[(Trained Models<br/>joblib)]
    end
    
    subgraph "Presentation Layer"
        Streamlit[Streamlit Dashboard]
        Overview[Overview Tab]
        Training[Training Tab]
        Insights[Insights Tab]
        Comments[Comment Analyzer]
        Plotly[Plotly Visualizations]
    end
    
    subgraph "AI Integration"
        GeminiAPI[Gemini API]
        ModelComparison[AI Model Comparison]
    end
    
    Reddit --> PRAW
    PRAW --> Fetch
    Fetch --> RawData
    RawData --> Preprocess
    Preprocess --> CleanedData
    CleanedData --> VADER
    VADER --> LabeledData
    LabeledData --> TFIDF
    TFIDF --> Models
    Models --> Pipeline
    Pipeline --> TrainedModels
    LabeledData --> Streamlit
    TrainedModels --> Streamlit
    Streamlit --> Overview
    Streamlit --> Training
    Streamlit --> Insights
    Streamlit --> Comments
    Streamlit --> Plotly
    Gemini --> GeminiAPI
    GeminiAPI --> ModelComparison
    ModelComparison --> Training
    Reddit --> Comments
```

## 2. System Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        User[User/Researcher]
        Browser[Web Browser]
    end
    
    subgraph "Application Layer"
        StreamlitApp[Streamlit Application<br/>dashboard.py]
        UI[User Interface Components]
        Sidebar[Sidebar Controls]
        Tabs[Tab Navigation]
    end
    
    subgraph "Business Logic Layer"
        DataLoader[Data Loading Module]
        ModelTrainer[Model Training Module]
        Visualizer[Visualization Module]
        CommentAnalyzer[Comment Analysis Module]
        AIHelper[AI Helper Module]
    end
    
    subgraph "Data Access Layer"
        FileSystem[File System]
        CSVReader[CSV Reader]
        JSONReader[JSON Reader]
        ModelLoader[Model Loader]
    end
    
    subgraph "Processing Layer"
        Preprocessor[Preprocessor Module<br/>clean_text, tokenize]
        Analyzer[Analyzer Module<br/>VADER, ML Pipeline]
        TFIDF[TF-IDF Vectorization]
    end
    
    subgraph "ML Framework Layer"
        ScikitLearn[Scikit-learn]
        Models[14 ML Models]
        Pipeline[Pipeline Builder]
        Metrics[Evaluation Metrics]
    end
    
    subgraph "External Services Layer"
        RedditAPI[Reddit API<br/>via PRAW]
        GeminiAPI[Google Gemini API]
    end
    
    subgraph "Storage Layer"
        RawStorage[(Raw Data<br/>reddit-posts.json)]
        ProcessedStorage[(Processed Data<br/>labeled-posts.csv)]
        ModelStorage[(Model Artifacts<br/>*.joblib)]
        OutputStorage[(Output Files<br/>PNG, CSV)]
    end
    
    User --> Browser
    Browser --> StreamlitApp
    StreamlitApp --> UI
    UI --> Sidebar
    UI --> Tabs
    Tabs --> DataLoader
    Tabs --> ModelTrainer
    Tabs --> Visualizer
    Tabs --> CommentAnalyzer
    Tabs --> AIHelper
    
    DataLoader --> FileSystem
    FileSystem --> CSVReader
    FileSystem --> JSONReader
    ModelTrainer --> ModelLoader
    ModelLoader --> ModelStorage
    
    ModelTrainer --> Analyzer
    Analyzer --> Preprocessor
    Analyzer --> TFIDF
    Analyzer --> ScikitLearn
    ScikitLearn --> Models
    Models --> Pipeline
    Pipeline --> Metrics
    
    CommentAnalyzer --> RedditAPI
    AIHelper --> GeminiAPI
    
    Preprocessor --> ProcessedStorage
    Analyzer --> ProcessedStorage
    Pipeline --> ModelStorage
    Visualizer --> OutputStorage
    
    CSVReader --> RawStorage
    JSONReader --> RawStorage
    CSVReader --> ProcessedStorage
```

## 3. Dataflow Diagram

```mermaid
flowchart LR
    Start([Start]) --> Collect[Collect Reddit Posts<br/>via PRAW]
    Collect --> Check{New Posts<br/>Exist?}
    Check -->|Yes| StoreRaw[Store Raw Data<br/>reddit-posts.json]
    Check -->|No| End1([End])
    
    StoreRaw --> Load[Load Raw Data]
    Load --> Clean[Clean Text<br/>Remove URLs, punctuation, numbers]
    Clean --> Tokenize[Tokenize & Remove<br/>Stopwords]
    Tokenize --> StoreClean[Store Cleaned Data<br/>cleaned-posts.csv]
    
    StoreClean --> VADER[Apply VADER<br/>Sentiment Analysis]
    VADER --> Label[Label Sentiment<br/>Positive/Negative/Neutral]
    Label --> StoreLabel[Store Labeled Data<br/>labeled-posts.csv]
    
    StoreLabel --> Split[Train/Test Split<br/>80/20]
    Split --> Vectorize[TF-IDF<br/>Vectorization]
    Vectorize --> Train[Train ML Models<br/>14 Different Models]
    Train --> Evaluate[Evaluate Models<br/>Accuracy, F1, Precision, Recall]
    Evaluate --> Save[Save Best Model<br/>models/*.joblib]
    
    Save --> Dashboard[Load into Dashboard]
    Dashboard --> Visualize[Generate Visualizations<br/>Charts, Word Clouds]
    Visualize --> Display[Display Results<br/>Streamlit UI]
    
    Display --> Export{Export<br/>Requested?}
    Export -->|Yes| ExportData[Export Data<br/>CSV, JSON, Excel, etc.]
    Export -->|No| End2([End])
    ExportData --> End2
    
    style Start fill:#90EE90
    style End1 fill:#FFB6C1
    style End2 fill:#FFB6C1
    style Collect fill:#87CEEB
    style Train fill:#DDA0DD
    style Dashboard fill:#F0E68C
```

## 4. Use Case Diagram

```mermaid
graph TB
    subgraph Actors
        User[User/Data Analyst]
        System[System]
        RedditAPI[Reddit API]
        GeminiAPI[Gemini API]
    end
    
    subgraph "Data Collection Use Cases"
        UC1[Collect Reddit Posts]
        UC2[Incremental Data Fetch]
        UC3[Duplicate Detection]
    end
    
    subgraph "Data Processing Use Cases"
        UC4[Clean Text Data]
        UC5[Apply VADER Sentiment]
        UC6[Label Data]
        UC7[Preprocess for ML]
    end
    
    subgraph "ML Training Use Cases"
        UC8[Select Models to Train]
        UC9[Train Multiple Models]
        UC10[Evaluate Model Performance]
        UC11[Compare Models]
        UC12[Save Trained Models]
        UC13[Load Saved Models]
    end
    
    subgraph "Visualization Use Cases"
        UC14[View Overview Dashboard]
        UC15[View Sentiment Distribution]
        UC16[View Time Trends]
        UC17[View Word Clouds]
        UC18[View Engagement Analysis]
        UC19[View Model Comparison Charts]
    end
    
    subgraph "Analysis Use Cases"
        UC20[Analyze Reddit Comments]
        UC21[Real-time Sentiment Analysis]
        UC22[Get AI Model Comparison]
        UC23[Export Data]
        UC24[Filter Data]
    end
    
    User --> UC1
    User --> UC8
    User --> UC9
    User --> UC14
    User --> UC15
    User --> UC16
    User --> UC17
    User --> UC18
    User --> UC19
    User --> UC20
    User --> UC21
    User --> UC22
    User --> UC23
    User --> UC24
    
    System --> UC2
    System --> UC3
    System --> UC4
    System --> UC5
    System --> UC6
    System --> UC7
    System --> UC10
    System --> UC11
    System --> UC12
    System --> UC13
    
    RedditAPI --> UC1
    RedditAPI --> UC20
    RedditAPI --> UC21
    
    GeminiAPI --> UC22
    
    UC1 --> UC2
    UC2 --> UC3
    UC3 --> UC4
    UC4 --> UC5
    UC5 --> UC6
    UC6 --> UC7
    UC7 --> UC8
    UC8 --> UC9
    UC9 --> UC10
    UC10 --> UC11
    UC11 --> UC12
    UC12 --> UC13
    UC13 --> UC14
    UC14 --> UC15
    UC14 --> UC16
    UC14 --> UC17
    UC14 --> UC18
    UC14 --> UC19
```

## 5. Activity Diagram - Model Training Workflow

```mermaid
flowchart TD
    Start([User Starts Training]) --> LoadData[Load Labeled Data]
    LoadData --> CheckData{Data Valid?}
    CheckData -->|No| Error1[Show Error Message]
    Error1 --> End1([End])
    CheckData -->|Yes| SelectModels[Select Models to Train]
    
    SelectModels --> InitProgress[Initialize Progress Bar]
    InitProgress --> LoopStart{More Models?}
    
    LoopStart -->|Yes| GetModel[Get Next Model]
    GetModel --> FilterData[Filter Neutral Sentiments]
    FilterData --> CheckFilter{Enough Data?}
    CheckFilter -->|No| SkipModel[Skip Model]
    CheckFilter -->|Yes| SplitData[Train/Test Split<br/>80/20]
    
    SplitData --> BuildPipeline[Build TF-IDF Pipeline]
    BuildPipeline --> TrainModel[Train Model]
    TrainModel --> Predict[Predict on Test Set]
    Predict --> CalculateMetrics[Calculate Metrics<br/>Accuracy, F1, Precision, Recall]
    CalculateMetrics --> SaveModel[Save Model<br/>models/*.joblib]
    SaveModel --> UpdateProgress[Update Progress Bar]
    
    UpdateProgress --> SkipModel
    SkipModel --> LoopStart
    
    LoopStart -->|No| SortResults[Sort Results by Accuracy]
    SortResults --> DisplayTable[Display Comparison Table]
    DisplayTable --> HighlightBest[Highlight Best Model]
    HighlightBest --> ShowCharts[Show Performance Charts]
    ShowCharts --> ShowConfusionMatrix[Show Confusion Matrix]
    ShowConfusionMatrix --> GenerateAI{Generate AI<br/>Comparison?}
    
    GenerateAI -->|Yes| CallGemini[Call Gemini API]
    CallGemini --> ParseResponse[Parse JSON Response]
    ParseResponse --> DisplayAITable[Display AI Comparison Table]
    DisplayAITable --> ExportOption{Export<br/>Results?}
    
    GenerateAI -->|No| ExportOption
    ExportOption -->|Yes| ExportCSV[Export to CSV]
    ExportOption -->|No| End2([End])
    ExportCSV --> End2
    
    style Start fill:#90EE90
    style End1 fill:#FFB6C1
    style End2 fill:#FFB6C1
    style TrainModel fill:#DDA0DD
    style CallGemini fill:#87CEEB
```

## 6. Activity Diagram - Real-time Comment Analysis

```mermaid
flowchart TD
    Start([User Enters Reddit URL]) --> ValidateURL{Valid<br/>Reddit URL?}
    ValidateURL -->|No| Error1[Show Invalid URL Error]
    Error1 --> End1([End])
    
    ValidateURL -->|Yes| Authenticate[Authenticate with Reddit<br/>via PRAW]
    Authenticate --> CheckAuth{Authentication<br/>Success?}
    CheckAuth -->|No| Error2[Show Auth Error]
    Error2 --> End1
    
    CheckAuth -->|Yes| FetchSubmission[Fetch Submission<br/>from Reddit]
    FetchSubmission --> ExpandComments[Expand All Comments<br/>replace_more]
    ExpandComments --> ExtractComments[Extract Comment Data<br/>author, text, score]
    
    ExtractComments --> CheckComments{Comments<br/>Found?}
    CheckComments -->|No| Error3[Show No Comments Message]
    Error3 --> End1
    
    CheckComments -->|Yes| CleanComments[Clean Comment Text]
    CleanComments --> AnalyzeSentiment[Apply VADER Sentiment<br/>to Each Comment]
    AnalyzeSentiment --> CreateDataFrame[Create DataFrame<br/>with Results]
    
    CreateDataFrame --> CalculateStats[Calculate Sentiment<br/>Statistics]
    CalculateStats --> DisplayPieChart[Display Pie Chart]
    DisplayPieChart --> DisplayKPIs[Display KPI Boxes<br/>Positive/Negative/Neutral]
    DisplayKPIs --> SortComments[Sort Comments by Score]
    SortComments --> DisplayTopPositive[Display Top 3<br/>Positive Comments]
    DisplayTopPositive --> DisplayTopNegative[Display Top 3<br/>Negative Comments]
    DisplayTopNegative --> ShowDataTable[Show Full Data Table<br/>in Expander]
    ShowDataTable --> End2([End])
    
    style Start fill:#90EE90
    style End1 fill:#FFB6C1
    style End2 fill:#FFB6C1
    style Authenticate fill:#87CEEB
    style AnalyzeSentiment fill:#DDA0DD
```

## 7. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant DataLoader
    participant Preprocessor
    participant Analyzer
    participant ModelTrainer
    participant Visualizer
    participant RedditAPI
    participant GeminiAPI
    
    User->>Dashboard: Launch Streamlit App
    Dashboard->>DataLoader: Load Processed Data
    DataLoader->>Dashboard: Return DataFrame
    
    User->>Dashboard: Select Models & Train
    Dashboard->>ModelTrainer: Train Models
    ModelTrainer->>Analyzer: Build Pipeline
    Analyzer->>Preprocessor: Clean Text
    Preprocessor-->>Analyzer: Cleaned Text
    Analyzer->>Analyzer: TF-IDF Vectorization
    Analyzer->>ModelTrainer: Train Model
    ModelTrainer-->>Dashboard: Model Results
    
    Dashboard->>GeminiAPI: Request AI Comparison
    GeminiAPI-->>Dashboard: Model Comparison JSON
    Dashboard->>Visualizer: Generate Charts
    Visualizer-->>Dashboard: Plotly Figures
    Dashboard->>User: Display Results
    
    User->>Dashboard: Enter Reddit URL
    Dashboard->>RedditAPI: Fetch Comments
    RedditAPI-->>Dashboard: Comment Data
    Dashboard->>Analyzer: Analyze Sentiment
    Analyzer->>Preprocessor: Clean Comments
    Preprocessor-->>Analyzer: Cleaned Comments
    Analyzer-->>Dashboard: Sentiment Labels
    Dashboard->>Visualizer: Create Visualizations
    Visualizer-->>Dashboard: Charts & KPIs
    Dashboard->>User: Display Analysis
```

---

## Diagram Descriptions

### 1. Proposed System Diagram
Shows the high-level system components and their relationships, including external APIs, data layers, ML components, and the presentation layer.

### 2. System Architecture Diagram
Details the technical architecture with layers: Client, Application, Business Logic, Data Access, Processing, ML Framework, External Services, and Storage.

### 3. Dataflow Diagram
Illustrates the complete data flow from collection through processing, training, visualization, and export.

### 4. Use Case Diagram
Defines all use cases and actors (User, System, Reddit API, Gemini API) and their interactions.

### 5. Activity Diagram - Model Training
Shows the step-by-step workflow for training multiple ML models, including error handling, progress tracking, and AI comparison integration.

### 6. Activity Diagram - Comment Analysis
Details the real-time comment analysis workflow from URL validation through sentiment analysis and visualization.

### 7. Component Interaction Diagram
Sequence diagram showing the interactions between components during typical operations.

---

*Note: These diagrams can be rendered in Markdown viewers that support Mermaid (GitHub, GitLab, many documentation tools) or converted to images using tools like Mermaid Live Editor or mermaid-cli.*

