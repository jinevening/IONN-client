gdb -ex=r --args ./build/examples/partitioning/classification.bin\
  models/bvlc_alexnet/deploy.prototxt \
  models/bvlc_alexnet/bvlc_alexnet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  models/bvlc_alexnet/prediction_model.txt \
  server_prediction_model.txt \
  80 \
  examples/images/cat.jpg \
  0.5 \
  all_at_once \
  time

