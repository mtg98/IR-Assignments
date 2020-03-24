--------------------------------------------------------------------------
----------Domain Specific Retrieval System ----------
--------------------------------------------------------------------------




Introduction
----------------

This Domain specific retrieval system is employed to search for certain fundamental rights, acts or laws in the form of queries and the most relevant constitutions are ranked on the basis of their respective TFxIDF scores. It should cater to the needs of lawyers, people working in legal firms at national and international levels, for students, researchers interested in studying and comparing about rights and duties of people across countries, their implementations through the study of their constitutions.For example, if a comparison has to be made in the right to equality, its implementations and principles practised across the globe.




Installation
---------------

1. nltk can be installed by using the command "pip install nltk". Visit: https://www.nltk.org/install.html
2. pyPDF can be installed by using the command "pip install pyPDF". Visit: https://pypi.org/project/pyPdf/
3. *Flask environment was also created to develop this web application.




Procedure
-------------

1. webpage_flask.py is the script that is run to start the server and start the web application at localhost. It renders search_page.html initially that provides the main interface to our domain search engine where queries that are to be searched are entered.
2. The entered search query on search_page is tokenised and the various tokens are matched against the files in the dataset by query_processing.py script being run in the backend. 
3. Following it, the data is reads data from the pickle files written when make_dictionary is executed and the respective TFxIDF scores are calculated.
4. On the above basis the documents are ranked(top 10) are dispayed on result_page.html and if no such query exists in the backend dataset, the a sorry message is displayed by the sorry_page.html.

