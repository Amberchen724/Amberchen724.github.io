---
layout:     post
title:      Identifying Hot Spots of "Emotional Geography" in New York City 
Subtitle:   This project presents two ways of identifying the ‘emotional geography’ of crowd behavior in public space.First is sentimental analysis on Twitter data, another is analyzing the data collected by using Electrodermal activity(EDA) sensor. 
date:       2019-04-19
author:     Yushi(Amber)Chen
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Geospatial Analysis
    - Sentiment Analysis  
    - Data Analysis
    
---

## Abstract: 

No matter one lives in urban space with high population density or lives in quieter environment at suburb area, people interact with the build environment they live in on daily bases. This project presents two ways of identifying the ‘emotional geography’ of crowd behavior in public space. By conducting sentiment analysis on twitter data in New York City, we are trying to investigate the “emotional geography” by using the tweets that collect from April 18th to April 25th. Furthermore, this project investigates individual pattern of “emotional geography” and exploring the same trend and pattern that the build environment may affect people’s emotion by using Electrodermal activity(EDA) sensor, which measures the skin conductance of volunteers. This information can be correlated with concrete urban planning requirements. Insights gained through the spatial analysis of our collected data can serve as a basis for understanding how people feel about the public space. 

## Methodology:
#### 1. NLP - Sentiment Analysis: 
 [Notebook](https://github.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/blob/master/sentimental_analysis_notebook.ipynb)
    
This project collected 24,000 tweets with geolocation from April 18th to April 25th in urban New York area, especially in Manhattan area, through twitter API. The data that collected from twitter include the information about twitter user name, twitter text, and geolocation. Sentiment Analysis is also known as opinion mining, it is a field within Nature Language Processing (NLP), which try to identify opinion and sentiment within text information and data. Beside identifying opinion, the system often extract attributes of the expression. This project used lexicon-based approach, which involves calculating orientation for a document from the semantic orientation of words or phrases in the document Among 24,000 tweets, there are number of 1956 tweets with “negative” sentiment label, 13191 tweets with “positive” sentiment label and 9115 tweets with “neutral” sentiment label. 
    
#### 2. Electrodermal activity(EDA) sensor:
[Notebook](https://github.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/blob/master/Data_Processing.ipynb)
    
The device we used to measure skin conductance is EMOTIONAL SENSOR - SKIN RESPONSE bought from Happy Electronics. It is a biofeedback sensor that can be attached to fingers or palm, measuring the change in skin conductivity at 5 Hz (5 sample points per second). The real-time skin conductivity data can be uploaded and saved when the sensor is connected with the computer by using the GSR studio software design for the sensor. According to the manual of the sensor, it could people's stress level. The increase in skin conductance from low level to high level means feeling stress while the decrease in skin conductance from high level to low level means feeling relax.


## Result:
1. I conducted hot spot analysis by using the twitter dataset that has sentiment analysis score. Hotspot analysis requires the presence of clustering within  the data. The Getis-Ord General G (Global) method will return value, including a z-score, and when analysed together will indicate if clustering is found in the data or not. After conducting the hotspot analysis, we created an interpolated surface that will effectively visualize the results of the hotspot analysis by using IDW ( inverse distance weighted ), which interpolate a raster surface from points. 

![png](https://raw.githubusercontent.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/master/HotSpotAnalysis.png)


From above two map for hotspot analysis, we can find out that both for the negative and positive hotspots’ clusters are located in lower Manhattan, especially in wall street area. For positive hotspot analysis, the cluster is more intensive in the wall street area, which means there are more positive tweets in the area and the distance between each tweets are smaller than other area. 

2. The hot spot map give us more details for where the significantly higher scores are. The red region indicates the cluster of stress point (the score in this region are significantly higher), while the blue region indicates the cluster of relax point (the score in this region are significantly lower). 
    
![png](https://raw.githubusercontent.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/master/Kernel%20Density.png)
 
we choose the representative stress region in different places and investigated each place by checking volunteer’s videos. Through the analysis of the video, we infer some possible factors that caused  people stress based on the feeling we sense and the objective environment in the video. 
    

![png](https://raw.githubusercontent.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/master/Female%20vs.%20Male.png)

![png](https://raw.githubusercontent.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/master/Weekend%20vs.%20Week.png)

![png](https://raw.githubusercontent.com/Amberchen724/Twitter-and-EDA_-Spatial-Analysis/master/Experience%20vs.%20No-experience.png)

## Discussion and Conclusion:  

In this project, through analysed  twitter data from New York City by sentiment analysis, we found out that the number of  positive tweets are larger than the number of negative tweets. We also found out the spatial pattern for those positive and negative tweets by conducting hotspot analysis, which indicated that people has different emotion in different spatial locations, even different “emotional geography” in same location. Furthermore, we have investigated tweets from a festival that happened in a public space where a crowd behavior might happened, we found out that, in festival setting, people’s emotion did not influenced by the crowd behavior according to 473 tweets we collected at that day.

The EDA analysis serves as another complement way to investigate the question. Through spatial analysis for 12 volunteers data, we identify the stress region in a small public space and used video recording to investigate the possible factors that causes people stress. Through the analysis between male and female, we identify that female in total is more emotional than male in feeling the positive and negative sides of public space. 

## Improvement:

There are several challenges  to make the result really convincing. The first is we might not have enough data. Only 12 volunteers data were captured and it is quite imbalance between weekdays and weekend, experienced volunteers and non-experienced volunteers. This could be mitigated by longer time of measurement and formally recruitment of volunteers from different backgrounds. Second, it is impossible to control all the factors in urban environment which cause problems to compare the result of different samples. Third, due to the machine error, the matching between location and emotional states was not perfect, which will affect the accuracy of our results.  More expensive and accurate devices could solve this problem. In 21 century,  when more and more people are moving into cities, besides building more infrastructures and applying more advanced technology, understanding the emotion reaction of people in urban space is valuable to construct a truly smart, humanitarian and enjoyable city. As Anderson & Smith (2001) said : “At particular times and in particular places, there are moments where lives are so explicitly lived through pain, bereavement, elation, anger, love and so on that the power of emotional relations cannot be ignored (and can readily be appreciated).”



```python

```
