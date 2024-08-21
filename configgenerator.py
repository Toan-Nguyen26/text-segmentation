import json

# // trong so do dai =  test do dai khac nhau
# // he so 
jsondata = {
#     "word2vecfile": "/home/omri/datasets/word2vec/GoogleNews-vectors-negative300.bin",
#     "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
#     "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r",

       # Ruunning on VSCode
    #   "word2vecfile": "./datasets/GoogleNews-vectors-negative300.bin",
    #   "wiki-test-50k": "./datasets/wiki_test_50",
    #   "wikidataset": "./datasets/wikidataset",
    #   "half-wikidataset": "./datasets/half-wikidataset",
    #   "snippets": "./datasets/snippets",

       # Ruunning on VSCode
      "word2vecfile": "./my_datasets/GoogleNews-vectors-negative300.bin",
      "wiki-test-50k": "./my_datasets/wiki_test_50",
      "wikidataset": "./my_datasets/wikidataset",
      "half-wikidataset": "./my_datasets/half-wikidataset",
      "snippets": "./my_datasets/snippets",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)

print("Configuration file generated successfully!")