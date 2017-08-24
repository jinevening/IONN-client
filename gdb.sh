gdb -ex=r --args ./build/examples/partitioning/classification.bin\
  models/bvlc_googlenet/deploy.prototxt \
  models/bvlc_googlenet/bvlc_googlenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  models/bvlc_googlenet/prediction_model.txt \
  server_prediction_model.txt \
  examples/images/cat.jpg

