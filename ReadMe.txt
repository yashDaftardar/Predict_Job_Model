README 

Instruction to run 3 files scraping.py extract_data.py, classify.py

Step 1- To Scape the data, you have to run scraping.py. To get job html file ypu have to provide job title and city name eg. scrape('Data Scientists','Cambridge') This will create a Data Scientist folder and inside that cambridge folder and inside you get html files.

Step 2- Once you have achieved desired Job title and city name data. extract_data.py will extract job description and store it in csv file filename as Jobs_Ads.csv  2 columns: <text>, <job title>. To extract the data you have to run extract_data.py by giving the folder name eg ad_data_scrape('Software Engineer') this will extract all the data from Software Engineer Folder

Step 3-  
1. Ensure Job_Ads.csv (scraped data file) is in the same directory as the classify.py program
2. Replace the TEST.csv with the input test file
3. Run the program
4. The predicted job titles are written to the Prediction.csv file.

Note: We checked the accuracy using 3 different models. As per our runs, LogisticRegression classifier gave the highest accuracy. You may uncomment and try to run the program using a different model.