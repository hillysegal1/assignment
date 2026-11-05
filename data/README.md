# Data

The full dataset is the Kaggle Titanic competition: <https://www.kaggle.com/competitions/titanic/data>.
Only `train.csv` is used. `train.py` and the notebook download it with `kagglehub` (see the main README
for the API token), so it is not stored in this repository.

Small samples for trying the app, taken from the held-out validation split:

| File | Rows | Contents |
|---|---|---|
| `sample.csv` | 25 | All Kaggle columns, including `Survived`: the app shows metrics and plots |
| `sample_unlabeled.csv` | 25 | The same passengers without `Survived`: the app shows predictions only |

Columns follow the Kaggle data dictionary: `PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch,
Ticket, Fare, Cabin, Embarked`.
