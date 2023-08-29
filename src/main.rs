#![recursion_limit = "256"]
#[macro_use]
extern crate log;
use std::borrow::BorrowMut;
//use std::borrow::BorrowMut;
use std::{fs::File, io::Write};
use std::io::BufReader;
use std::collections::{VecDeque,HashMap,HashSet};
//use std::process::id;
use serde::{Deserialize, Serialize};
//use serde_json::value::Index;
use std::cell::RefCell;
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;
//use std::env;
use std::io::BufWriter;
use std::time::Instant;
use indexmap::IndexMap;
use indicatif::{ProgressBar,ProgressStyle};
use std::sync::{Arc, Mutex};
use std::thread;
use std::io;

/// A struct representing the node data
/// * 'row' - the row index of the node
/// * 'col' - the column index of the node
/// * 'accum' - the total accumulation value of the node
/// * 'elev' - the elevation of the node
/// * 'neighbors' - a vector containing the row and column indices of the node neighbor/children
/// * 'is_explored' - a flag used when traversing nodes
#[derive(Serialize, Deserialize, Debug,Clone)]
struct Node {
    row: u32,
    col: u32,
    accum: f64,
    elev:f64,
    neighbors:Vec<Vec<u32>>,

    #[serde(default)]
    is_explored:bool,
}

/// A struct containing relevant drainage node relative to a starting node
/// * 'node_id' - ID of the cell where the search started from
/// * 'closest' - Closest drainage cell ids to the given node ID
#[derive(Serialize, Deserialize, Debug,Clone)]
struct Drainage {
    node_id:u32,
    closest: Vec<u32>,
}

/// Returns the ID of a node given its row and column indices
/// * 'row' - the row index of the node
/// * 'col' - the column index of the node
/// * 'cols' - the number of columns in the digital elevation model
fn id_hash(row:u32,col:u32,cols:u32) -> u32 {
    return row*cols + col;
}

/// Returns the row and column indices a node given its id
/// * 'node_id' -  the ID of the node
/// * 'cols' - the number of clolumns in the digital elevation model
fn reverse_hash(node_id:u32,cols:u32) -> (u32,u32){
    let row = node_id/cols; //integer division
    let col = node_id - row*cols;
    return (row,col)
}


/// Returns the *Manhattan distance* (abs(dx) + abs(dy)) between
/// **two node_ids**.
/// 
/// Used to sort the drainage cells according to their proximity to the 
/// starting cell.
/// 
/// * 'id_a' - the ID of the first node
/// * 'id_b' - the ID of the second node
fn manhattan(id_a:u32,id_b:u32,cols:u32) -> u32{
    let (ra,ca) = reverse_hash(id_a, cols);
    let (rb,cb) = reverse_hash(id_b,cols);
    let dr = (ra as i32) - (rb as i32);
    let dc = (ca as i32)  - (cb as i32);
    return (dr.abs() + dc.abs()) as u32;
}

/// Searches for all drainage nodes hat can be traversed from a given **start_index**. The
/// function employs a **breadth-first-search** as facilitated by the Node neighbors/children.
/// 
/// If a node is a drainage, it returns an empty vector.
/// 
/// * 'start_index' - the node id where the search for drainage nodes starts
/// * 'data' - a reference to the network data comprised of all nodes in the digital elevation model
/// * 'threshold' - the value that discriminates a drainage node from non-drainage node
/// * 'cols' - the numner of columns in the digital elevation model

fn search_drainage(start_index:u32,
                    data:&Arc<Vec<Node>>,
                    closest_drainage:&Arc<Mutex<Vec<Drainage>>>,
                    threshold:f64,
                    rows:u32,
                    cols:u32,
                    max_drainage:usize,
                    pb:Arc<Mutex<ProgressBar>>) {
    let data_len = rows * cols;
    //let mut explored:Vec<u32> = Vec::with_capacity(2000);
    //accessing directly from Node.is_explored is faster try to mitigate with multithreading
    let mut explored:HashSet<u32> = HashSet::new(); 


    let mut stack:VecDeque<u32> = VecDeque::with_capacity(1000);
    let mut drainage_ids:Vec<u32> = Vec::with_capacity(1000);
    //explored.push(start_index);
    explored.insert(start_index);
    
    stack.push_back(start_index);
    //data[start_index as usize].borrow_mut().is_explored = true;

    //if the starting cell is a river, return nothing
    let start_accum = data[start_index as usize].accum;
    if start_accum >= threshold {
        //write empty drainage struct
        
        let drainage = Drainage{node_id: start_index,closest:drainage_ids};
        let mut drainage_guard = closest_drainage.lock().unwrap();
        drainage_guard[start_index as usize] = drainage;
    } else {
        while stack.len() > 0 {
            let node_index = stack.pop_front().unwrap();
            let neighbor_locs = data[node_index as usize].to_owned().neighbors;
            for loc in neighbor_locs.iter(){
                let row = loc[0];
                let col = loc[1];

                let neighbor_id = id_hash(row, col, cols);

                //if the index is within the data set
                //if let Some(element)  = data.get(neighbor_id as usize){
                    
                    //let neighbor_is_explored =  element.borrow().is_explored;
                    
                    
                    if !explored.contains(&neighbor_id) && (neighbor_id < data_len) {
                    //if !neighbor_is_explored {
                        let neighbor_accum = data[neighbor_id as usize].accum;
                        if neighbor_accum >= threshold {
                            drainage_ids.push(neighbor_id);
                        } else {
                            stack.push_back(neighbor_id);
                        }
                        
                        //element.borrow_mut().is_explored = true;
                        //explored.push(neighbor_id);
                        explored.insert(neighbor_id);
                    }
                    
                //}

            }
        }

        drainage_ids.sort_by(|val_a,val_b| {
                    let dist_a = manhattan(*val_a,start_index, cols);
                    let dist_b  = manhattan(*val_b,start_index, cols);
                    //least to greatest
                    dist_a.cmp(&dist_b)
                });
                


        let drainage_len = drainage_ids.len();
        let num_to_store:usize = drainage_len.min(max_drainage);
        let drainage = Drainage{node_id: start_index,closest:drainage_ids[0..num_to_store].to_vec()};
        let mut drainage_guard = closest_drainage.lock().unwrap();
        drainage_guard[start_index as usize] = drainage;
    }




    /*//reset explored nodes
    for explored_index in explored.iter() {
        if let Some(element)  = data.get(explored_index.to_owned() as usize){
            element.borrow_mut().is_explored = false;
        }
    }*/


    pb.lock().unwrap().inc(1);
    //return drainage;

}


/// Returns the IDs of cells traversed from the given **node_id** to an **end node**.
/// 
///  The threshold value catergorizes the cells according to their accumulation values.
/// 
/// The function is recursive as it is based from a **depth-first-search** algorithm but without keeping track
/// of the explored nodes.
/// 
/// * 'node_id' - the ID of the node where the search starts from
/// * 'end' - the ID of the node where the search ends at
/// * 'cols' - the numner of columns in the digital elevation model
/// * 'threshold' - the value that discriminates a drainage node from non-drainage node
/// * 'path' - a reference to a vector that contains the path traversed from the start to the end node
/// * 'result' - a reference to a vector that would contain the possible paths generated from the search
/// * 'data' - a reference to the network data comprised of all nodes in the digital elevation model
fn dfs_modified(
    node_id:u32,
    end:u32,
    rows:u32,
    cols:u32,
    threshold: f64,
    max_path_length:usize,
    path:&mut Vec<u32>,
    visited:&mut Vec<u32>,
    result:&mut Vec<Vec<u32>>,
    data: &Arc<Vec<Node>>) {
    
    visited.push(node_id.clone());
    let data_len = rows * cols;
    let mut path = path.to_owned();
    //println!("START:{node_id},END:{end}");
    //println!("{:?}",visited);
    path.push(node_id);
    if node_id == end {
        result.push(path.clone());
    } else {
        let neighbor_locs = data[node_id as usize].to_owned().neighbors;

        for loc in neighbor_locs.iter() {
            let row = loc[0];
            let col = loc[1];
            let neighbor_id = &id_hash(row, col, cols);
            //println!("{neighbor_id}");
            //check if the neighbor is within bounds
            if (neighbor_id < &data_len) && !visited.contains(neighbor_id) && (path.len() < max_path_length) {
                if !path.contains(neighbor_id) {
                    dfs_modified(*neighbor_id,
                                end,
                                rows,
                                cols,
                                threshold,
                                max_path_length,
                                &mut path,
                                visited,
                                result,
                                data);
                }
                /*let neighbor_accum = data[*neighbor_id as usize].borrow_mut().to_owned().accum;
                if (*neighbor_id != end) && (path.len() > 2) {
                    //recursion in here
                    if (neighbor_accum < threshold) && (!path.contains(neighbor_id)) {
                        dfs_modified(*neighbor_id,
                                    end,
                                    rows,
                                    cols,
                                    threshold,
                                    &mut path,
                                    result,
                                    data);
                    }
                } else {
                    //recursion in here (if the neighbor is the end or if the path consists of the neighbor and end)
                    //the only difference is we're not checking if neighbor is a drainage cell and we're not checking
                    //if the cell is already the end
                    if !path.contains(neighbor_id) {
                        dfs_modified(*neighbor_id,
                                    end,
                                    rows,
                                    cols,
                                    threshold,
                                    &mut path,
                                    result,
                                    data);
                    }
                }*/
            } else {
                warn!("ID: {neighbor_id} -> ({row},{col}) is out of bounds.");
            }
        }
    }

    //remove the current node so it does not duplicate
    path.pop();
}


/// Returns all possible path from a given **starting node** to an **end node**.
/// 
/// The returned value are vectors whose first and last element are the 
/// starting and ending nodes respectively.
///  
/// Nodes are represented by their ids.
/// * 'start' - the ID of the node where the search starts from
/// * 'end' - the ID of the node where the search ends at
/// * 'cols' - the numner of columns in the digital elevation model
/// * 'threshold' - the value that discriminates a drainage node from non-drainage node
/// * 'data' - a reference to the network data comprised of all nodes in the digital elevation model
fn find_all_paths(
    start:u32,
    end:u32,
    rows:u32,
    cols:u32,
    threshold:f64,
    max_path_length:usize,
    data: &Arc<Vec<Node>>,
    results: Arc<Mutex<Vec<Vec<Vec<u32>>>>>,
    pb: Arc<Mutex<ProgressBar>>) {
    
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(100);
    let mut path: Vec<u32> = Vec::with_capacity(100);
    let mut visited: Vec<u32> = Vec::with_capacity(100);
    dfs_modified(start,
         end,
         rows,
         cols,
         threshold,
         max_path_length,
         &mut path,
         &mut visited,
         &mut result,
         data);
    
    pb.lock().unwrap().inc(1);

    let mut guard = results.lock().unwrap();
    guard[start as usize].extend(result);
}

/// This function returns the id of the chosen drainage cell passing through the different
/// candidate paths.
/// 
/// Alpha controls the *path length bias* over the *accumulation bias* on the basis that
/// the shorter the path the better the candidate and the higher average accumulation the better
fn select_paths(paths:&Vec<Vec<u32>>,data:&Vec<RefCell<Node>>,alpha:f64) ->u32 {
    if paths.len() == 0 {
        panic!("Tries to select path from an empty path list.");
    }
    
    let paths = paths.clone();

    let mut average_accums: HashMap<usize,f64> = HashMap::new();
    for (index,path) in paths.iter().enumerate() {
        let mut sum:f64 = 0.0;
        for node_id in path {
            let accum = data[*node_id as usize].borrow().to_owned().accum;
            sum += accum;
        }
        average_accums.insert(index, sum/(path.len() as f64));
    }

    
    let path_lengths:HashMap<usize,f64> = paths
                    .iter().enumerate()
                    .map(|(index,path)|{
                        (index,path.len() as f64)
                    })
                    .collect();

    // get the score based on the alpha bias
    let mut scores:Vec<f64> = Vec::with_capacity(paths.len());
    for i in 0..paths.len(){
        let path_lengths_sum:f64 = path_lengths.values().sum();
        let current_path_length = path_lengths.get(&i).unwrap();
        let ave_accum_sum:f64 = average_accums.values().sum();
        let current_ave_accum = average_accums.get(&i).unwrap();

        let path_score:f64 = 1.0 - (current_path_length/path_lengths_sum);
        let accum_score:f64 = current_ave_accum/ave_accum_sum;
        let score = alpha * path_score + (1.0 - alpha) * accum_score;
        scores.push(score);
    }

    
    //get maximum score
    let max_score = scores.iter().fold(f64::NEG_INFINITY, |max, &x| max.max(x));

    //filter paths with score equal to the max_score
    let mut filtered_paths:Vec<Vec<u32>> = paths
                        .iter()
                        .enumerate()
                        .filter(|(index,_)|{
                            scores[*index] == max_score
                        })
                        .map(|(_,path)| {
                            path.clone()
                        })
                        .collect();

    //no ties, return index of drainage cell            
    if filtered_paths.len() == 1 {
        return *filtered_paths[0].last().unwrap();
    } else {

        //resolve ties by choosing drainage with higher elevation for a conservative hand value
        filtered_paths.sort_by(|path1,path2|{
            let drainage1 = path1.last().unwrap();
            let drainage2 = path2.last().unwrap();
            
            //sort in descending order of elevation
            drainage2.cmp(&drainage1)
        });

        return *filtered_paths[0].last().unwrap();
    }
}



fn main() {
    env_logger::init();

    let network_file_path = "./accumulations/cells.json";
    let rows: u32 = 1047;//5;
    let cols: u32 = 1613;//6;
    let data_len = (rows * cols) as usize;


    //number of closest drainage (manhattan distance not path length) to consider
    let max_drainage:usize = 5;

    //terminate dfs for path searching when current length is over this
    let max_path_length:usize = 30;

    //path length bias over average accumulation per path (range is 0 to 1) for path selection
    let alpha = 0.9;


    println!("Loading network data.");
    let start = Instant::now();
    let file = File::open(&network_file_path).unwrap();
    let reader = BufReader::new(file);
    let deserialized: IndexMap<u32,Node> = serde_json::from_reader(reader).unwrap();


    //let deserialized: IndexMap<u32,Node> = serde_json::from_reader(file).expect("Failed to deserialize JSON file.");
    let vec_data:Vec<Node> = deserialized.values().cloned().collect();
    let data: Vec<RefCell<Node>> = vec_data
                .iter()
                .cloned()
                .map(|item|RefCell::new(item))
                .collect();
    
    let arc_data: Arc<Vec<Node>> = Arc::new(vec_data);


    let elapsed = start.elapsed();
    println!("Graph data successfuly serialized. Searching for closest drainage...");
    println!("Elapsed time: {:.2?}",elapsed);


    //create progress bar
    let pb = ProgressBar::new(data_len as u64);
    let sty = ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-");
    pb.set_style(sty.clone());
    let pb:Arc<Mutex<ProgressBar>> = Arc::new(Mutex::new(pb));


    
    let mut closest_drainage:Vec<Drainage> = Vec::with_capacity(data_len);
    for i in 0..data_len {
        let empty_drainage = Drainage{node_id: i as u32, closest: Vec::new()};
        closest_drainage.push(empty_drainage);
    }
    let closest_drainage:Arc<Mutex<Vec<Drainage>>> = Arc::new(Mutex::new(closest_drainage));


    let start = Instant::now();
    let mut num_proccessed = 0;
    let collect_every = 1000;
    let mut handles = Vec::new();
    let drainage_threshold:f64 = 1000.0;//5.0;
    for r in 0..rows {
        for c in 0..cols {
            let clone_pb = Arc::clone(&pb);
            let clone_data = Arc::clone(&arc_data);
            let clone_closest_drainage = Arc::clone(&closest_drainage);

            let start_index = id_hash(r,c,cols);
            let handle = thread::spawn( move || {
                        search_drainage(start_index,
                                &clone_data,
                                &clone_closest_drainage,
                                drainage_threshold,
                                rows,
                                cols,
                                max_drainage,
                                clone_pb);
                        });
            handles.push(handle);
            
            //pop handles at the set interval
            if num_proccessed % collect_every == 0 {
                for _ in 0..handles.len() {
                    handles.pop().unwrap().join().unwrap();
                }
            }

            num_proccessed += 1;
        }
    }


    //pop remaining handles
    for _ in 0..handles.len() {
        handles.pop().unwrap().join().unwrap();
    }


    pb.lock().unwrap().finish();
    let elapsed = start.elapsed();
    println!("Nearby drainage nodes identified. Finding all paths to each drainage node...");
    println!("Elapsed time: {:.2?}\n",elapsed);


    //extract data out from Arc<Mutex<Vec<Drainage>>>
    let mut closest_drainage = closest_drainage.lock().unwrap().clone();


    //store the paths per node to its connected drainage cells
    let paths_to_drainage: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new();data_len]));


    //arc mutex progress bar
    let task_count:Vec<u64> = closest_drainage
                                .iter()
                                .cloned()
                                .map(|drain|{
                                    drain.closest.len() as u64
                                })
                                .collect();
    /*let num_closest_zeros:Vec<u64> = num_closest
                                .iter()
                                .cloned()
                                .filter(|num|{
                                    *num == 0
                                })
                                .collect();
    let non_empty_tasks:u64 = num_closest.iter().sum();
    //let empty_tasks: u64 = num_closest_zeros.len() as u64;*/

    //let num_tasks:u64 = non_empty_tasks + empty_tasks;

    let num_tasks = task_count.iter().sum();



    let pb = ProgressBar::new(num_tasks);
    let sty = ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-");
    pb.set_style(sty.clone());
    let pb:Arc<Mutex<ProgressBar>> = Arc::new(Mutex::new(pb));
    
    
    


    let mut num_proccessed = 0;
    let collect_every = 1000;
    let mut handles = Vec::new();
    for start_node in 0..data_len {
        let start_node_accum = data[start_node].borrow().accum;
        if start_node_accum < drainage_threshold {
            
            let end_nodes = closest_drainage[start_node].borrow_mut().clone().closest;

            //only process if the node is connected to a river cell
            if end_nodes.len() > 0 {
                for end_node in end_nodes {
                    let clone_data = Arc::clone(&arc_data);
                    //data.clone_into(&mut clone_data);
                    let clone_paths_to_drainage = Arc::clone(&paths_to_drainage);
                    let clone_pb = Arc::clone(&pb);
                    let handle = thread::spawn(move|| {
                        find_all_paths(start_node as u32,
                             end_node,
                             rows,
                             cols,
                             drainage_threshold,
                             max_path_length,
                             &clone_data,
                             clone_paths_to_drainage,
                             clone_pb);
                    });
                    handles.push(handle);

                    if num_proccessed % collect_every == 0 {
                        for _ in 0..handles.len() {
                            handles.pop().unwrap().join().unwrap();
                        }
                    }

                    num_proccessed += 1
                }
            }
        }
    }
    
    //finish progress bar
    pb.lock().unwrap().finish();

    //pop remaining handles
    for _ in 0..handles.len() {
        handles.pop().unwrap().join().unwrap();
    }

    

    //create HAND array
    //represent drainage cells as -1
    //represent no drainage path as -2
    let mut hand: Array2<f64> = Array2::from_elem((rows as usize,cols as usize),-1.0);
    let final_paths = paths_to_drainage.lock().unwrap().clone();
    for (node_id,paths) in final_paths.iter().enumerate() {
        
        let node = data[node_id].borrow_mut().to_owned();
        let row = node.row;
        let col = node.col;

        if paths.len() == 0 {
            hand[[row as usize, col as usize]] = -2.0;
            //println!("{node_id} - Closest Drainage: None");
        } else {
            let closest_drainage:u32 = select_paths(paths, &data, alpha);
            let elev = node.elev;
            let drain_elev = data[closest_drainage as usize].borrow_mut().to_owned().elev;
            let mut hand_value = elev - drain_elev;
            if hand_value < 0.0 {
                hand_value = 0.0
            }

            hand[[row as usize, col as usize]] = hand_value;

            //println!("{node_id} - Closest Drainage: {closest_drainage}, HAND: {hand_value}");
        }
    }










    let writer = BufWriter::new(File::create("./accumulations/hand.npy").unwrap());
    match hand.write_npy(writer){
        Ok(_) => {println!("{} was written.","./accumulations/hand.npy")},
        Err(err) => {println!("An error occured while writing {}. See below:\n{}","./accumulations/hand.npy",err)},
    };

    let json_string = serde_json::to_string(&closest_drainage).unwrap();
    let mut file = File::create("./accumulations/closest_neighbor.json").expect("Unable to create closest_neighbor.json");
    match file.write_all(json_string.as_bytes()) {
        Ok(_) => {println!("{} was written.","./accumulations/closest_neighbor.json")},
        Err(err) => {println!("An error occured while writing {}. See below:\n{}","./accumulations/closest_neighbor.json",err)},
    };


    println!("Enter any to exit.");
    let mut _input = String::new();
    io::stdin().read_line(&mut _input).expect("Failed to read line.");
    print!("Program exited.");
}
