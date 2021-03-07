# Dathathon Interbank - Top %54 (294 teams)
> _[Kaggle competition](https://www.kaggle.com/c/interbank20/overview)_

## My experience through this
I think it was the first time that I got into the compete alone but with some tools that I've learned in this long knowledge path.

I'm not comfortable enough with the result to be honest. I did many attempts with differents points of views but it wasn't worth. Also I realize I should
have organize a way to test new ideas. Sometimes I got lost amid my own commits and ideas, at the end of the day, it took more time that it should.

Sadly, I wasted time trying to set a new local computer with linux that I have in my house. I thought It would be useful to optimize hyperparameters, but was a bad
idea configure it after the competition have started. Even though, after the compete finished, I configure it and now is ready for the next compete.

On the other hand, I think now I masterize my knowledge git and pycharm due te limit time and sometimes the pressure of upload my submissions.
Those tools helped me very well. Also some theoretical aspects I know deeply better now. I read a lot about lightgbm hyperparameters to tuning it,
I tried to add more to optimize them but with just 3 I got better results.

## Technical aspects
I used Lightgbm with five folds to train the model with parameter-optimization. I also tried with xgboost but it had lower performance than Lightgbm.

All the variables were obfuscated , I tried to decode the most important of them , I put many days here without any successful outcome :)

From the beanchmark 3 (code gives by the organizer) I used the "makeCt" function where by key values itinerate with agg functions to find Max,Min,Len and avg 
At the end I tried undersampling with a worse result than the previos one. I think I didn't select correctly the random rows

## About the dataset:
- I didn't use all the databases because it decreased the performance (Or I didn't find the correct way to make it useful)
- I did mix some features with brute force, creating a new dataset. After that I did feature selection adding some random variables between 1 and 0 and with the feature
importance I could realize which of them were good

*Still A lot to improve, but im working on it.*

