{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fmodern\fcharset0 CourierNewPSMT;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww14040\viewh9640\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # If I want to use a different metric (1h, or 3h rain, for instance)\
\
Step 1: Recreate 
\f1 df_roads
\f0  (adjust 
\f1 ENV_VARIABLES
\f0 )
\f1 .\
	
\f0 ->
\f1  analyzeInputData.py\
\

\f0 Step 2: Adjust 
\f1 ENV_FEATURES
\f0  and 
\f1 input_file.\
	
\f0 ->
\f1  prepareData.py 
\f0 ->
\f1  create_training_validation_test_data()\
\

\f0 Step 3: Adjust
\f1  ENV_FEATURES.\
	
\f0 ->
\f1  testModel.py }