Design and prototype a complete modern full-stack web application UI called:

“PrivacyML Sentinel”

Subtitle:
“Privacy-Preserving Fraud Detection using Differential Privacy”

Purpose:
This is an enterprise-style cybersecurity and machine-learning decision-support platform. It allows a Data Steward to securely upload sensitive tabular datasets, configure Differential Privacy, run privacy-preserving fraud detection, evaluate model utility and privacy-attack risk, compare different privacy budgets ε, and identify an appropriate privacy–utility trade-off.

IMPORTANT:
Do NOT design this as a generic AI dashboard.
The application should visually communicate:
- Information security
- Data protection
- Differential Privacy
- Privacy risk analysis
- Machine learning
- Privacy–utility decision support

Use a professional, clean, trustworthy cybersecurity SaaS aesthetic.

--------------------------------------------------
DESIGN SYSTEM
--------------------------------------------------

Overall style:
- Modern enterprise cybersecurity SaaS
- Minimal and professional
- Clean white/light background
- Dark navy text
- Subtle blue/indigo accents
- Green for secure/success states
- Amber for warnings
- Red only for genuine security risks
- Rounded cards with subtle borders
- Moderate corner radius
- Soft shadows
- Excellent whitespace
- Clear hierarchy
- Avoid excessive gradients
- Avoid flashy neon cyberpunk styling
- Avoid unnecessary illustrations
- Avoid excessive glassmorphism

Typography:
- Use Inter or a similar modern sans-serif font.
- Strong bold headings.
- Medium-weight section titles.
- Highly readable body text.
- Monospace font only for technical values such as ε, Δf, API status, etc.

Icons:
Use simple professional line icons:
- Shield
- Lock
- Database
- Upload
- Settings
- Activity
- Chart
- Brain/ML
- User
- File
- Security
- Alert
- Check
- Download
- History

--------------------------------------------------
APPLICATION STRUCTURE
--------------------------------------------------

Create these pages/screens:

1. Login / Authentication
2. Dashboard
3. Dataset Upload
4. Privacy Configuration
5. Secure Processing
6. Model Analysis
7. Privacy Analysis
8. Privacy–Utility Explorer
9. Experiment History
10. Reports
11. Profile / Account Settings

Create a persistent authenticated application shell with:
- Left sidebar navigation
- Top header
- User profile menu
- Notifications
- Main content area

--------------------------------------------------
1. LOGIN / AUTHENTICATION PAGE
--------------------------------------------------

Create a polished login page.

Layout:
Two-column desktop layout.

LEFT:
Branding and product explanation.

Display:
“PrivacyML Sentinel”

Subtitle:
“Secure. Private. Explainable.”

Short description:
“Privacy-preserving machine learning for sensitive financial and tabular data.”

Show 3 small benefits:
- Encrypted Data Storage
- Differential Privacy
- Privacy–Utility Analysis

RIGHT:
Login card.

Fields:
- Email
- Password

Controls:
- Remember me checkbox
- Forgot password?
- Sign In button

Also include:
“Don't have an account? Create account”

Authentication states:
- Invalid credentials
- Loading
- Successful login

Create a separate Sign Up screen/modal with:
- Full name
- Email
- Password
- Confirm password
- Create account button

Security messaging:
“Your sensitive datasets are protected using encrypted storage and privacy-preserving processing.”

Do not claim specific security guarantees that are not implemented.

--------------------------------------------------
2. MAIN DASHBOARD
--------------------------------------------------

After login, show the main dashboard.

Header:
“Privacy ML Dashboard”

Greeting:
“Welcome back, Data Steward”

Top action:
“+ New Analysis”

Top summary cards:

Dataset
- Current dataset name
- Number of records
- Number of features

Privacy Budget
- Current ε
- DP status: Enabled

Model
- Logistic Regression
- Current model status

Security
- Encryption: Active
- Secure Vault: Active

Main dashboard sections:

A. CURRENT ANALYSIS

Display:
Dataset: Credit Card Fraud Detection
Records: 284,807
Features: 30
Target: Class
Privacy Budget: ε = 1.0
Status: Analysis Complete

B. MODEL PERFORMANCE

Five metric cards:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Use realistic placeholder values only.
Clearly label them as example/demo data.

C. PRIVACY RISK

Two cards:
- Membership Inference Risk
- Reconstruction Risk

Each should show:
Risk level
Risk score
Short explanation

D. PRIVACY–UTILITY TRADE-OFF

Large interactive chart.

X-axis:
Privacy Budget ε

Y-axis:
Model Utility / Privacy Risk

Allow toggling:
- Accuracy
- F1-Score
- ROC-AUC
- MIA Risk
- Reconstruction Risk

E. RECOMMENDATION CARD

Title:
“Recommended Privacy Budget”

Example:
“ε = 1.0”

But clearly label it:
“Based on current experiment”

Show:
Privacy level
Utility level
Attack risk

Button:
“View Analysis”

Do not imply that ε = 1 is universally optimal.

--------------------------------------------------
3. DATASET UPLOAD PAGE
--------------------------------------------------

Page title:
“Dataset Upload”

Large drag-and-drop upload area.

Text:
“Upload a CSV dataset for privacy-preserving analysis.”

Buttons:
- Browse Files
- Upload Dataset

After upload show:

Dataset Summary:
- File name
- File size
- Number of records
- Number of columns
- Target column
- Missing values

Dataset preview table.

For the benchmark dataset show:
Credit Card Fraud Detection
284,807 records
31 attributes
30 input features
Target: Class

Add:
“Data is encrypted before persistent storage.”

Add a small security status:
“Secure Upload”

CTA:
“Continue to Privacy Configuration”

--------------------------------------------------
4. PRIVACY CONFIGURATION PAGE
--------------------------------------------------

Page title:
“Privacy Configuration”

Main component:
Differential Privacy configuration card.

Large ε slider.

Label:
“Privacy Budget (ε)”

Slider range:
0.1 → 10

Show:
Current ε = 1.0

Below slider explain:

Lower ε
→ More noise
→ Stronger privacy
→ Potentially lower model utility

Higher ε
→ Less noise
→ Better model utility
→ Comparatively weaker privacy

Create three preset buttons:
- Strong Privacy
- Balanced
- High Utility

Do NOT state that these presets are guaranteed privacy levels.

Show technical information:

Differential Privacy Mechanism:
Laplace Mechanism

Global Sensitivity:
Δf = 1.0
“Derived after feature normalization”

Noise relationship:
Noise Scale = Δf / ε

Add expandable section:
“Advanced Configuration”

Show:
- Mechanism
- Sensitivity
- Privacy budget
- Data preprocessing status

CTA:
“Run Secure Analysis”

--------------------------------------------------
5. SECURE PROCESSING PAGE
--------------------------------------------------

Create a visual processing pipeline.

Title:
“Secure Analysis in Progress”

Show sequential steps:

1. Dataset Received
2. Dataset Encrypted
3. Stored in Secure Vault
4. Decrypted Temporarily in RAM
5. Data Cleaning
6. Min-Max Scaling
7. Differential Privacy Injection
8. Logistic Regression Training
9. Utility Evaluation
10. Privacy Attack Evaluation
11. Privacy–Utility Analysis
12. Results Generated

Each stage should have:
- Status icon
- Description
- Progress state

States:
Completed
In Progress
Pending

Important security visualization:

SECURITY STATUS

Data at Rest:
Encrypted

Persistent Plaintext Copy:
None

Processing:
Transient Memory

Differential Privacy:
Enabled

Privacy Budget:
ε = 1.0

After completion:
“Analysis Complete”

CTA:
“View Results”

--------------------------------------------------
6. MODEL ANALYSIS PAGE
--------------------------------------------------

Title:
“Model Performance Analysis”

Subtitle:
“Evaluate predictive utility on the privacy-preserved dataset.”

Top metric cards:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Add a confusion matrix section:

                 Predicted
              Legitimate  Fraud
Actual Legitimate
Actual Fraud

Use clear labels:
True Positive
True Negative
False Positive
False Negative

Add charts:
- ROC Curve
- Precision–Recall Curve

Add an ε comparison chart:
“Model Performance vs Privacy Budget”

Allow user to select:
Accuracy
Precision
Recall
F1
ROC-AUC

Include a note:
“Model performance may change as the privacy budget changes.”

--------------------------------------------------
7. PRIVACY ANALYSIS PAGE
--------------------------------------------------

Title:
“Privacy Attack Analysis”

Subtitle:
“Evaluate resistance to privacy attacks.”

Create two major cards.

A. MEMBERSHIP INFERENCE ATTACK

Show:
What it means:
“Attempts to determine whether a record was part of the model’s training data.”

Metrics:
- Attack Success Rate
- MIA Risk
- Baseline vs DP-protected model

Comparison:

Without DP
Attack Success: [value]

With DP
Attack Success: [value]

Show:
“Lower attack success indicates stronger resistance.”

B. RECONSTRUCTION ATTACK

Show:
What it means:
“Attempts to infer or reconstruct sensitive information from exposed model-related information.”

Metrics:
- Reconstruction Error
- Reconstruction Risk

Comparison:
Baseline vs DP-protected system

Important:
Clearly distinguish experimental results from illustrative/demo values.

Add a privacy risk summary:

Overall Privacy Risk
LOW / MEDIUM / HIGH

--------------------------------------------------
8. PRIVACY–UTILITY EXPLORER
--------------------------------------------------

This should be the flagship page of the application.

Title:
“Privacy–Utility Explorer”

Subtitle:
“Explore how privacy protection affects model utility.”

Allow user to select multiple ε values:

☑ 0.1
☑ 0.5
☑ 1
☑ 5
☑ 10

Button:
“Run Comparison”

After comparison show a large table:

ε
Accuracy
Precision
Recall
F1
ROC-AUC
MIA Risk
Reconstruction Risk

Add multiple interactive charts:

1. ε vs Accuracy
2. ε vs F1-Score
3. ε vs Privacy Risk
4. ε vs MIA Risk
5. Privacy–Utility Trade-off

Allow chart metric selection.

Highlight the currently selected ε.

Create an “Optimal Trade-off” panel.

Show:
Recommended ε
Reason
Privacy level
Utility level
Attack risk

Recommendation must be based on measured results and a clearly defined decision criterion.

--------------------------------------------------
9. EXPERIMENT HISTORY
--------------------------------------------------

Title:
“Experiment History”

Table columns:

Experiment ID
Dataset
ε
Date
Accuracy
F1
Privacy Risk
Status
Action

Example rows:
EXP-001
CreditCard.csv
ε = 0.1
Completed

EXP-002
CreditCard.csv
ε = 1
Completed

EXP-003
CreditCard.csv
ε = 10
Completed

Actions:
- View
- Compare
- Delete

Add filters:
- Dataset
- ε
- Date
- Status

--------------------------------------------------
10. REPORTS PAGE
--------------------------------------------------

Title:
“Analysis Reports”

Show generated reports as cards/table.

Each report contains:
- Dataset
- Privacy budget
- Model
- Utility metrics
- Privacy metrics
- Date

Actions:
- View Report
- Download Report
- Compare

Add:
“Generate New Report”

Report should contain:
1. Dataset information
2. Security configuration
3. Privacy configuration
4. Model metrics
5. Privacy attack metrics
6. Privacy–utility analysis
7. Recommended ε
8. Summary

--------------------------------------------------
11. PROFILE / ACCOUNT SETTINGS
--------------------------------------------------

Show:
- Name
- Email
- Role: Data Steward
- Account created date

Security settings:
- Change password
- Sign out
- Active sessions

Do not add unnecessary personal data.

--------------------------------------------------
SIDEBAR NAVIGATION
--------------------------------------------------

Persistent sidebar:

Logo:
PrivacyML Sentinel

Navigation:

Dashboard
Dataset Upload
Privacy Configuration
Secure Processing
Model Analysis
Privacy Analysis
Privacy–Utility Explorer
Experiment History
Reports

Divider

Settings
Help
Logout

Use icons beside each item.

Highlight active page.

--------------------------------------------------
TOP HEADER
--------------------------------------------------

Top bar should contain:

Current dataset:
“Credit Card Fraud Detection”

Security status:
“Protected”

DP status:
“Enabled”

Notifications icon

User avatar:
“Data Steward”

Dropdown:
Profile
Settings
Logout

--------------------------------------------------
IMPORTANT UX BEHAVIOR
--------------------------------------------------

The frontend should visually communicate the complete architecture:

Data Steward
↓
React Frontend
↓
FastAPI Backend
↓
Security Layer
↓
Secure Vault
↓
Transient RAM
↓
Preprocessing
↓
Differential Privacy
↓
Private Dataset
↓
Logistic Regression
↓
Utility Analyzer
↓
Privacy Analyzer
↓
Privacy–Utility Analyzer
↓
React Dashboard

Make this architecture visible through the Secure Processing screen.

--------------------------------------------------
IMPORTANT SECURITY UX
--------------------------------------------------

Use security status indicators throughout the application.

Example:

SECURE
Encrypted at Rest

SECURE
Transient Processing

PROTECTED
Differential Privacy Enabled

ANALYSIS
Privacy Risk Evaluated

Never display raw sensitive dataset contents unnecessarily.

Use masked/summarized previews where appropriate.

--------------------------------------------------
RESPONSIVE DESIGN
--------------------------------------------------

Create:
- Desktop version as the primary design
- Tablet adaptation
- Mobile adaptation

Desktop dashboard should use a 12-column grid.

Sidebar:
240–260 px

Top navigation:
64–72 px

Cards:
Consistent spacing and hierarchy.

Charts should be large enough to read clearly.

--------------------------------------------------
COMPONENT LIBRARY
--------------------------------------------------

Create reusable Figma components for:

Buttons
Inputs
Dropdowns
Slider
Cards
Metric Cards
Status Badges
Security Badges
Tables
Charts
Modal
Toast Notifications
Progress Steps
Sidebar
Top Navigation
Tabs
Tooltips
Risk Indicators
Empty States
Loading States
Error States
Confirmation Dialogs

Create variants for:
Default
Hover
Active
Disabled
Loading
Success
Warning
Error

--------------------------------------------------
FINAL VISUAL GOAL
--------------------------------------------------

The finished application should feel like a real enterprise cybersecurity product that could be presented during an academic project review.

The most visually important concepts should be:

1. Secure data handling
2. Differential Privacy
3. Configurable ε
4. Membership Inference Risk
5. Reconstruction Risk
6. Model Performance
7. Privacy–Utility Trade-off
8. Recommended ε
9. Interactive decision support

The interface should make it immediately obvious that this is not merely a fraud detection application.

It is a:
“Privacy-Preserving Machine Learning Decision-Support System.”

Use realistic but clearly illustrative demo data in charts and metrics.
Do not claim experimental results unless explicitly provided.