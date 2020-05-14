import json
import numpy as np
import boto3
import csv
from datetime import datetime
from decimal import Decimal


def lambda_handler(event, context):
    # TODO implement
    
    s3 = boto3.resource(u's3')
    
    bucket = s3.Bucket(u'appfactree-ml-experiments-s3-bucket')
    
    dynamo = boto3.resource("dynamodb").Table("mlChallengeResultsSubmission")
    
    nowDateTime = str(datetime.now())
    
    http_Method = event['requestContext']['http']['method']
    
    returnData = {}
    metricsData = {}

    if http_Method == "GET":
        returnData["errorMessage"] = "GET not supported"
    elif http_Method == "POST":
        body = json.loads(event['body'])
        challengeName = body['challengeName']
        challengeType = body['challengeType']
        userID = body['userID']
        submissionsData = body['submissionsData']
    
        submissionLines = submissionsData.splitlines()
        loop = 0
        for line in submissionLines:
            data = line.split(',')
            if loop == 0:
                subarr = np.array([np.int(data[0]),np.int(data[1])])
            else:
                subarr = np.vstack((subarr,np.array([np.int(data[0]),np.int(data[1])])))
            loop = loop+1
        
        obj = bucket.Object(key=u'testdatafiles/'+challengeName+'.csv')
        
        response = obj.get()
        
        data =  response[u'Body'].read().decode('utf-8').splitlines() #response[u'Body'].read().split()
        
        lines = csv.reader(data)
        
        headers = next(lines)
    
        loop = 0
        for line in lines:
            if loop == 0:
                actarr = np.array([np.int(line[0]),np.int(line[1])])
            else:
                actarr = np.vstack((actarr,np.array([np.int(line[0]),np.int(line[1])])))
            loop = loop+1
        
        actarr = actarr[actarr[:, 0].argsort()]
        subarr = subarr[subarr[:, 0].argsort()]
        
        returnData['challengeName']= challengeName
        returnData['challengeType']= challengeType
        returnData['userID']= userID

        
        if challengeType == 'binaryclassification':

                falsePositive = list(map(lambda x,y:(x[0]==y[0] and x[1]==1 and y[1]==0),subarr,actarr))
                falsePositiveCount = int(np.sum(falsePositive))
                
                truePositive = list(map(lambda x,y:(x[0]==y[0] and x[1]==1 and y[1]==1),subarr,actarr))
                truePositiveCount = int(np.sum(truePositive))
                
                falseNegative = list(map(lambda x,y:(x[0]==y[0] and x[1]==0 and y[1]==1),subarr,actarr))
                falseNegativeCount = int(np.sum(falseNegative))
                
                trueNegative = list(map(lambda x,y:(x[0]==y[0] and x[1]==0 and y[1]==0),subarr,actarr))
                trueNegativeCount = int(np.sum(trueNegative)) 

                metricsData['truePositive'] = truePositiveCount
                metricsData['falsePositive'] = falsePositiveCount
                metricsData['falseNegative'] = falseNegativeCount
                metricsData['trueNegative'] = trueNegativeCount
                
                if truePositiveCount>0 or falsePositiveCount>0:
                    precision = round(truePositiveCount/(truePositiveCount+falsePositiveCount),2)
                else:
                    precision = 0
                
                if truePositiveCount>0 or falseNegativeCount>0:    
                    recall = round(truePositiveCount/(truePositiveCount+falseNegativeCount),2)
                else:
                    recall = 0
                
                metricsData['precision'] = Decimal(str(precision))
                metricsData['recall'] = Decimal(str(recall))
                
                accuracy = (truePositiveCount+trueNegativeCount)/(truePositiveCount+falsePositiveCount+trueNegativeCount+falseNegativeCount)
                metricsData['accuracy'] = Decimal(str(round(accuracy,2)))
                
                if precision>0 or recall>0:
                    f1score = 2*precision*recall/(precision+recall)
                else:
                    f1score = 0
                    
                metricsData['f1score'] = Decimal(str(round(f1score,2)))
                
        elif challengeType == 'multiclassification':
                dataComp = (actarr==subarr)
                matchCount = len([elem for elem in dataComp if elem[0]==True and elem[1]==True])
                
                metricsData['matchCount'] = matchCount
                
                matchPercentage = matchCount/actarr.shape[0]
                
                metricsData['matchPercentage'] = Decimal(str(round(matchPercentage,2)))
                
        elif challengeType == 'regression':
                dataComp = np.sqrt((actarr-subarr)*(actarr-subarr))
                rmse = np.sum(dataComp[:,1])
                metricsData['rmse']= Decimal(str(rmse))
         

        returnData.update(metricsData)
        
        for key in returnData:
            returnData[key] = str(returnData[key])
        
        
        intDateTime = int(nowDateTime.replace("-","").replace(":","").replace(".","").replace(" ",""))
        PartitionKey = str(userID)+"-"+challengeName+"-"+str(intDateTime)
      
      
        response = dynamo.put_item(Item={"ChallengeName": challengeName,
        "ChallengeType": challengeType,
        "UserId": userID,
        "F1Score":metricsData['f1score'],
        "AllMetrics": metricsData,
        "InsertDateTime": intDateTime,
        "PartitionKey" : PartitionKey
        })
                
        print(returnData)
        
    
    return {
        'statusCode': 200,
        'body': json.dumps(returnData)
    }
