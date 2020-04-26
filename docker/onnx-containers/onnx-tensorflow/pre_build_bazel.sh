"
git clone https://github.com/tensorflow/tensorflow ./tensorflow-git && \
cd ./tensorflow-git && \
bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel build tensorflow/tools/graph_transforms:transform_graph && \
cp bazel-bin/tensorflow/tools/graph_transforms/summarize_graph ./ && \
cp bazel-bin/tensorflow/tools/graph_transforms/transform_graph ./ && \
cd ../../ && \
rm -rf ./tensorflow-git
"