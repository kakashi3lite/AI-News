-- Market Signal D1 seed (idempotent, 2026-08-05)
INSERT OR IGNORE INTO "Source" ("name","url","type","category","reliabilityScore","isActive","createdAt") VALUES
('Wall Street Journal','https://feeds.a.dj.com/rss/RSSMarketsMain.xml','rss','business',0.96,true,CURRENT_TIMESTAMP),
('Bloomberg','https://feeds.bloomberg.com/markets/news.rss','rss','business',0.96,true,CURRENT_TIMESTAMP),
('Financial Times','https://www.ft.com/rss/home','rss','business',0.95,true,CURRENT_TIMESTAMP),
('BBC News','http://feeds.bbci.co.uk/news/rss.xml','rss','general',0.94,true,CURRENT_TIMESTAMP),
('NPR','https://feeds.npr.org/1001/rss.xml','rss','general',0.93,true,CURRENT_TIMESTAMP),
('The Guardian','https://www.theguardian.com/world/rss','rss','world',0.92,true,CURRENT_TIMESTAMP),
('CNN','http://rss.cnn.com/rss/edition.rss','rss','general',0.88,true,CURRENT_TIMESTAMP),
('The Verge','https://www.theverge.com/rss/index.xml','rss','technology',0.87,true,CURRENT_TIMESTAMP),
('TechCrunch','https://techcrunch.com/feed/','rss','technology',0.85,true,CURRENT_TIMESTAMP),
('Hacker News','https://hnrss.org/frontpage','rss','technology',0.8,true,CURRENT_TIMESTAMP),
('Google News','https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en','rss','general',0.9,true,CURRENT_TIMESTAMP),
('Google News Business','https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en','rss','business',0.9,true,CURRENT_TIMESTAMP),
('Google News Tech','https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en','rss','technology',0.9,true,CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO "WatchlistItem" ("name","aliases","keywords","feeds","category","createdAt","updatedAt") VALUES
('Microsoft','["MSFT","MS"]','["viva","teams","copilot","azure","linkedin"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Google','["Alphabet","GOOGL","GOOG","DeepMind"]','["gemini","workspace","chrome","android","waymo"]','[]','technology',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Workday','["WDAY"]','["human capital","hr software","workday"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('SAP','[]','["successfactors","sap","erp","s/4hana"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Lattice','[]','["performance reviews","people ops","lattice"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Culture Amp','["CultureAmp"]','["employee engagement","culture amp"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Personio','[]','["personio","hr software","recruiting"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('HiBob','["Bob"]','["hibob","hr platform","people management"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Staffbase','[]','["employee communications","internal comms","staffbase"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Simpplr','[]','["simpplr","intranet","employee experience"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Firstup','[]','["firstup","employee comms","internal communications"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Slack','["Salesforce"]','["slack","salesforce","chatops"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP),
('Guru','[]','["guru","knowledge management","ai search"]','[]','business',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP);
