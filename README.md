## PhenoTrack3D: an automatic high-throughput phenotyping pipeline to track maize organs over time

PhenoTrack3D can be used to track leaf organs over time, in a time-series of 3D maize plant reconstructions. 
Such 3D reconstructions can be obtained from sets of RGB images with the Phenomenal pipeline (https://github.com/openalea/phenomenal)

### Maintainers

* Daviet Benoit (benoit.daviet@inrae.fr)
* Fournier Christian (christian.fournier@inrae.fr)
* Pradal Christophe (christophe.pradal@cirad.fr)

### License

Our code is released under **Cecill-C** (https://cecill.info/licences/Licence_CeCILL_V1.1-US.txt) licence. 
See LICENSE file for details.

### Install

install dependencies with conda:

	conda create -n phenotrack -c conda-forge -c openalea3 openalea.phenomenal skan=0.10 pytest numpy skimage
	conda activate phenotrack

Clone repo and run setup

    git clone https://github.com/openalea/phenotrack3d.git
    cd phenotrack3d


### References

If you use PhenoTrack3D to your research, cite:

Daviet, B., Fernandez, R., Cabrera-Bosquet, L. et al. PhenoTrack3D: an automatic high-throughput phenotyping pipeline to track maize organs over time. Plant Methods 18, 130 (2022). https://doi.org/10.1186/s13007-022-00961-4
    
```latex
@article {daviet22,
	author = {Daviet, Benoit and Fernandez, Romain and Cabrera-Bosquet, Lloren{\c c} and Pradal, Christophe and Fournier, Christian},
	title = {PhenoTrack3D: an automatic high-throughput phenotyping pipeline to track maize organs over time},
	elocation-id = @article{daviet2022phenotrack3d,
	title={PhenoTrack3D: an automatic high-throughput phenotyping pipeline to track maize organs over time},
	author={Daviet, Benoit and Fernandez, Romain and Cabrera-Bosquet, Lloren{\c{c}} and Pradal, Christophe and Fournier, Christian},
	journal={Plant Methods},
	volume={18},
	number={1},
	pages={1--14},
	year={2022},
	publisher={Springer}
}

```
