import json

# // trong so do dai =  test do dai khac nhau
# // he so 
jsondata = {
#     "word2vecfile": "/home/omri/datasets/word2vec/GoogleNews-vectors-negative300.bin",
#     "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
#     "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r",
      "word2vecfile": "./datasets/GoogleNews-vectors-negative300.bin",
      "wiki-test-50k": "./datasets/wiki_test_50",
      "wikidataset": "./datasets/wikidataset",
      "half-wikidataset": "./datasets/half-wikidataset",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)

print("Configuration file generated successfully!")