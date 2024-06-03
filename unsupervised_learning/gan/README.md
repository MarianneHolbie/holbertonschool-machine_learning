# GAN

This project about GANs is organized as follows :

* The description contains a crash course about GANs. The fundamentals ideas behind GANs are exposed as fast as possible.
* [Task 0](./0-simple_gan.py) : the class `Simple_GAN` : as they were introduced, GANs are a game played by two adversary players.
* [Task 1](./1-wgan_clip.py) : the class `WGAN_clip` : later it appeared that in fact the two players can collaborate.
* [Task 2](./2-wgan_gp.py) : the class `WGANGP` : a more natural version of the latter, which outperforms `SimpleGANs` and `WGAN_clips` in higher dimensions, as will be illustrated in the main program.
* [Task 3](./3-generate_faces.py) : convolutional generators and discriminators : to work with pictures we need to use convolutional neural networks.
* [Task 4](4-wgan_gp.py) : our own face generator : we will use a `WGAN_GP` model to produce faces of persons that don’t exist.
* Appendix : at the end of task 4 there is a short digression explaining why GANs are superior to the older generators of fake pictures that were based on PCA.