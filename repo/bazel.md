# Install `bazel`: with `bazelisk`


Release Page: [github link](https://github.com/bazelbuild/bazelisk/releases)

```sh
sudo wget https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-linux-amd64 \
  -O /usr/local/bin/bazel && \
  sudo chmod +x /usr/local/bin/bazel
```

or

```sh
sudo npm install -g @bazel/bazelisk
sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel
```
