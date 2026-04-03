Submission Format

For every instance in the dataset, submission files should contain two columns: id and Accept.

The file should contain a header and have the following format:

id,Accept
bae908d5352,0
9260b4c0f25,1
2c4e5bbee21,1
feb453f1ab5,0
aa32073d39f,0
e0617187aef,0



Submission

Before the challenge deadline:

    you can submit every day your test file with the predictions at most 100 times
    before the deadline, remember to mark the two submissions that you select as the best ones for calculating the final leaderboard

After the kaggle submission deadline, use moodle to submit:

    The notebooks
    A pdf file containing a table with one column per group member and one row per notebook, indicating the authoring of the notebooks.
    Individual and group assignments

How to submit (code example)

You should load the training dataset for training your algorithm:

url = "data/train.csv" 
df = pd.read_csv(url)
df

You should also load the test dataset without labels so that you can make the predictions

url= "data/test_nolabel.csv"
df_test = pd.read_csv(url)
df_test

If you apply some preprocessing steps to the training dataset, remember that you will have to apply the same steps to the test dataset. For example, if you add a new column, your model will expect that column. Repeat the same steps with the test dataset, but you cannot eliminate any row.

Suppose that df_test_clean is the transformed df_test after preprocessing it.

You can get the submission file quickly, as shown below.

X = df_test_clean[features].values
df_test_clean['Accept'] = model.predict(X)

*To get the csv file in the right format*
df_test_clean['Accept'] = df_test_clean['Accept'].astype(np.int)
df_test_clean.to_csv('my-model.csv', columns=['id','Accept'], index = False)


--------- VS Code configuration

Connect to
Configuration

# Add this to your settings.json:
{
    "servers": {
        "kaggle": {
            "url": "https://www.kaggle.com/mcp",
            "type": "http"
        }
    },
}

Authentication

# Call the MCP server's `authorize` tool.

# Or, for token authentication:
{
    "servers": {
        "kaggle": {
            "url": "https://www.kaggle.com/mcp",
            "type": "http",
            "headers" : {
                "authorization": "Bearer <YOUR_TOKEN>"
            }
        }
    },
}
    
NOTE: my token is in the variable KAGGLE_API_TOKEN of this computer

# If you don't already have a token, go to Settings > Generate New Token > Copy.

Usage

# To upload a submission, prompt your client to use the
# "mcp_kaggle_start_competition_submission_upload" tool. 
# Then, use "kaggle_mcp_submit_to_competition" to submit it to the competition.

