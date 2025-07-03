FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# FROM ubuntu:latest

ARG NVIM_CONFIG_REPO
ARG DEBIAN_FRONTEND=noninteractive

# Repository mgmt
RUN apt-get update && apt-get install -y \
    software-properties-common

# Login
RUN apt-get update && apt-get install -y \
    passwd \
    sudo \
    login \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash dev \
    && passwd -d dev && echo "dev:dev" | chpasswd \
    && usermod -aG sudo dev

# Nvim and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    python3 \
    python3-pip \
    unzip \
    ripgrep \
    fd-find \
    npm \
    golang \
    cargo \
    luarocks \
    ruby \
    ruby-dev \
    php \
    composer \
    cmake \
    default-jdk \
    sqlite3 \
    ghostscript \
    graphviz \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Clangd setup (respecting mason organization)
RUN apt-get update && apt-get install -y clangd jq

RUN cd /tmp && \
    wget https://github.com/neovim/neovim/releases/download/v0.11.2/nvim-linux-arm64.tar.gz && \
    tar -xzf nvim-linux-arm64.tar.gz && \
    mv nvim-linux-arm64 /opt/nvim && \
    ln -s /opt/nvim/bin/nvim /usr/local/bin/nvim && \
    rm nvim-linux-arm64.tar.gz

USER dev

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash - && \
    echo 'export NVM_DIR="$HOME/.nvm"' >> /home/dev/.bashrc && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm' >> /home/dev/.bashrc && \
    echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion' >> /home/dev/.bashrc && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    nvm install --lts

RUN mkdir -p ~/.local/share/nvim/mason/packages/clangd/mason-schemas && \
    cd ~/.local/share/nvim/mason/packages/clangd && \
    curl https://raw.githubusercontent.com/clangd/vscode-clangd/master/package.json | \
    jq .contributes.configuration > mason-schemas/lsp.json && \
    echo '{"schema_version":"1.1","primary_source":{"type":"local"},"name":"clangd","links":{"share":{"mason-schemas/lsp/clangd.json":"mason-schemas/lsp.json"}}}' > mason-receipt.json


RUN git clone "$NVIM_CONFIG_REPO" /home/dev/.config/nvim
RUN echo "cloned nvim config from $NVIM_CONFIG_REPO"

WORKDIR /workspace
CMD ["/bin/bash"] 
