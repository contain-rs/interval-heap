var searchIndex = {};
searchIndex['interval_heap'] = {"items":[[0,"","interval_heap","A double-ended priority queue implemented with an interval heap.",null,null],[3,"IntervalHeap","","A double-ended priority queue implemented with an interval heap.",null,null],[3,"Iter","","An iterator over an `IntervalHeap` in arbitrary order.",null,null],[3,"IntoIter","","A consuming iterator over an `IntervalHeap` in arbitrary order.",null,null],[11,"clone","","",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"intervalheap"}}],[11,"default","","",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"intervalheap"}}],[11,"new","","Returns an empty heap ordered according to the natural order of its items.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"intervalheap"}}],[11,"with_capacity","","Returns an empty heap with the given capacity and ordered according to the\nnatural order of its items.",0,{"inputs":[{"name":"intervalheap"},{"name":"usize"}],"output":{"name":"intervalheap"}}],[11,"from","","Returns a heap containing all the items of the given vector and ordered\naccording to the natural order of its items.",0,{"inputs":[{"name":"intervalheap"},{"name":"vec"}],"output":{"name":"intervalheap"}}],[11,"with_comparator","","Returns an empty heap ordered according to the given comparator.",0,{"inputs":[{"name":"intervalheap"},{"name":"c"}],"output":{"name":"intervalheap"}}],[11,"with_capacity_and_comparator","","Returns an empty heap with the given capacity and ordered according to the given\ncomparator.",0,{"inputs":[{"name":"intervalheap"},{"name":"usize"},{"name":"c"}],"output":{"name":"intervalheap"}}],[11,"from_vec_and_comparator","","Returns a heap containing all the items of the given vector and ordered\naccording to the given comparator.",0,{"inputs":[{"name":"intervalheap"},{"name":"vec"},{"name":"c"}],"output":{"name":"intervalheap"}}],[11,"iter","","Returns an iterator visiting all items in the heap in arbitrary order.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"iter"}}],[11,"min","","Returns a reference to the smallest item in the heap.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"option"}}],[11,"max","","Returns a reference to the greatest item in the heap.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"option"}}],[11,"min_max","","Returns references to the smallest and greatest items in the heap.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"option"}}],[11,"capacity","","Returns the number of items the heap can hold without reallocation.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"usize"}}],[11,"reserve_exact","","Reserves the minimum capacity for exactly `additional` more items to be inserted into the\nheap.",0,{"inputs":[{"name":"intervalheap"},{"name":"usize"}],"output":null}],[11,"reserve","","Reserves capacity for at least `additional` more items to be inserted into the heap.",0,{"inputs":[{"name":"intervalheap"},{"name":"usize"}],"output":null}],[11,"shrink_to_fit","","Discards as much additional capacity from the heap as possible.",0,{"inputs":[{"name":"intervalheap"}],"output":null}],[11,"pop_min","","Removes the smallest item from the heap and returns it.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"option"}}],[11,"pop_max","","Removes the greatest item from the heap and returns it.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"option"}}],[11,"push","","Pushes an item onto the heap.",0,{"inputs":[{"name":"intervalheap"},{"name":"t"}],"output":null}],[11,"into_vec","","Consumes the heap and returns its items as a vector in arbitrary order.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"vec"}}],[11,"into_sorted_vec","","Consumes the heap and returns its items as a vector in sorted (ascending) order.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"vec"}}],[11,"len","","Returns the number of items in the heap.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"usize"}}],[11,"is_empty","","Returns `true` if the heap contains no items.",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"bool"}}],[11,"clear","","Removes all items from the heap.",0,{"inputs":[{"name":"intervalheap"}],"output":null}],[11,"fmt","","",0,{"inputs":[{"name":"intervalheap"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"from_iter","","",0,{"inputs":[{"name":"intervalheap"},{"name":"i"}],"output":{"name":"intervalheap"}}],[11,"extend","","",0,{"inputs":[{"name":"intervalheap"},{"name":"i"}],"output":null}],[11,"extend","","",0,{"inputs":[{"name":"intervalheap"},{"name":"i"}],"output":null}],[11,"clone","","",1,{"inputs":[{"name":"iter"}],"output":{"name":"iter"}}],[11,"next","","",1,{"inputs":[{"name":"iter"}],"output":{"name":"option"}}],[11,"size_hint","","",1,null],[11,"next_back","","",1,{"inputs":[{"name":"iter"}],"output":{"name":"option"}}],[11,"next","","",2,{"inputs":[{"name":"intoiter"}],"output":{"name":"option"}}],[11,"size_hint","","",2,null],[11,"next_back","","",2,{"inputs":[{"name":"intoiter"}],"output":{"name":"option"}}],[11,"into_iter","","",0,{"inputs":[{"name":"intervalheap"}],"output":{"name":"intoiter"}}]],"paths":[[3,"IntervalHeap"],[3,"Iter"],[3,"IntoIter"]]};
initSearch(searchIndex);