# keras-extra
Extra Layers that I have added to Keras
Commented out layers mean that they have been added to the Keras master branch
Copyright Aran Nayebi, 2015
anayebi@stanford.edu

If you already have Keras installed, for this to work on your current installation, please do the following:
1. Upgrade to the newest version of Keras (since some layers may have been added from here that are now commented out):
    sudo pip install --upgrade git+git://github.com/fchollet/keras.git
or, if you don't have super user access, just run:
    pip install --upgrade git+git://github.com/fchollet/keras.git --user

2. Add this file to your Keras installation in the layers directory (keras/layers/)

3. Now, to use any layer, just run:
    from keras.layers.extra import layername
