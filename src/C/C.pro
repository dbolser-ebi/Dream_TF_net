QT += core
QT -= gui

CONFIG += c++11

TARGET = C
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

LIBS += \
       -lboost_system\
       -lboost_filesystem\
       -lboost_iostreams\
       -lgsl\
       -lgslcblas\

