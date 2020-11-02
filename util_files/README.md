RENDERING:\
There is differenet type of rtendering supported in this repository.
Simple rendering with curves and lines and other representation is paths.

If your data represented as paths than chose 
data_representation='paths' and look at use case at graphics/graphics.py#L199 

If your data represented as curves or lines than use 
data_representation='vahe’
and as input give dict 
 {PT_LINE: [[x0, y0, x1, y1, w], …], PT_BEZIER: [[x0, y0, …, w], …]}.
 
Loss functions:\
To train your network for vectorization there is several loss functions presented in  
loss_functions folder supervised  and semi-supoervised loss functions:
supervised-
semi-supervised-

 