#!/bin/bash

echo "Starting container cleanup before saving to image..."

# Remove temporary files
echo "Cleaning temporary files..."
rm -rf /tmp/* /var/tmp/* 2>/dev/null || true

# Clear package manager caches
echo "Cleaning package manager caches..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get clean && rm -rf /var/lib/apt/lists/*
fi

if command -v yum >/dev/null 2>&1; then
    yum clean all
fi

if command -v apk >/dev/null 2>&1; then
    apk cache clean
fi

# Clear pip cache if Python is installed
echo "Cleaning Python pip caches..."
if command -v pip >/dev/null 2>&1; then
    pip cache purge 2>/dev/null || true
fi

if command -v pip3 >/dev/null 2>&1; then
    pip3 cache purge 2>/dev/null || true
fi

# Remove log files
echo "Cleaning log files..."
find /var/log -type f -name "*.log" -delete 2>/dev/null || true
find /var/log -type f -name "*.log.*" -delete 2>/dev/null || true

# Clear bash history
echo "Clearing shell history..."
rm -f /root/.bash_history ~/.bash_history 2>/dev/null || true

# Remove any core dumps
echo "Removing core dumps..."
find / -name "core.*" -type f -delete 2>/dev/null || true

# Clear any cached files
echo "Clearing cache directories..."
find /var/cache -type f -delete 2>/dev/null || true

# Remove Python bytecode files
echo "Removing Python bytecode files..."
find / -name "*.pyc" -delete 2>/dev/null || true
find / -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true

# Remove man pages and documentation (optional - uncomment if needed)
# echo "Removing documentation files..."
# rm -rf /usr/share/man/* /usr/share/doc/* 2>/dev/null || true

# Remove locales except en_US (optional - uncomment if needed)
# echo "Removing unused locales..."
# find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name "en_US*" -exec rm -rf {} \; 2>/dev/null || true

echo "Container cleanup completed successfully!"
