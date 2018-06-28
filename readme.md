
# What's your Fake Tweeter avatar !?!

## Finding similarity between your tweet (or text) and fake tweets presented in the NBC dataset.

- Input [system argument] :
    - Any text (preferably a tweet on the same topics as being discussed in these fake tweets)
        : string


- Output :
    - Details of you Fake Tweeter avatar
    - Most used hashtags and most retweeted tweets from the Fake Tweeter


- Process :
    - Every word in the input string is tokenized and encoded to a vector using 50 Dimensional glove encodings
    - Next, we take a centroid for the entire input so we get a 50 Dimensional representation of the input
    - Finally, predict the Fake Tweeter most similar to the input text
    - Lastly, gather details about the Fake Tweeter from the merged dataset and present in order


- How to use :
    - Run fake_tweeter.py with the input text as an argument for the first run :
        '''python3 fake_tweeter.py "North Korean Leader Kim Jong Un just stated that the 'Nuclear Button is on his desk at all times'."'''

    - From second iteration, the program waits for the user input and input can be directly entered there without running the file again
