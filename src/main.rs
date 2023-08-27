#![recursion_limit = "256"]
use std::borrow::BorrowMut;
//use std::borrow::BorrowMut;
use std::{fs::File, io::Write};
//use std::io::Read;
use std::collections::VecDeque;
//use std::process::id;
use serde::{Deserialize, Serialize};
//use serde_json::value::Index;
use std::cell::RefCell;
use ndarray::Array2;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyExt};
//use std::env;
use std::io::BufWriter;
use std::time::Instant;
use indexmap::IndexMap;
use indicatif::{ProgressBar,ProgressStyle};
use std::num;


/// A struct representing the node data
/// * 'row' - the row index of the node
/// * 'col' - the column index of the node
/// * 'accum' - the total accumulation value of the node
/// * 'elev' - the elevation of the node
/// * 'neighbors' - a vector containing the row and column indices of the node neighbor/children
/// * 'parents' - a vector containing the row and column indices of the node parents
/// * 'is_explored' - a flag used when traversing nodes
#[derive(Serialize, Deserialize, Debug,Clone)]
struct Node {
    row: u32,
    col: u32,
    accum: f64,
    elev:f64,
    neighbors:Vec<Vec<u32>>,
    parents:Vec<Vec<u32>>,

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
fn search_drainage(start_index:u32,data:&mut Vec<RefCell<Node>>,threshold:f64,cols:u32) ->Vec<u32> {
    
    let mut explored:Vec<u32> = Vec::with_capacity(2000);
    let mut stack:VecDeque<u32> = VecDeque::with_capacity(1000);
    let mut drainage:Vec<u32> = Vec::with_capacity(1000);
    explored.push(start_index);
    stack.push_back(start_index);
    data[start_index as usize].borrow_mut().is_explored = true;

    //if the starting cell is a river, return nothing
    let start_accum = data[start_index as usize].borrow().accum;
    if start_accum >= threshold {
        return drainage;
    }

    while stack.len() > 0 {
        let node_index = stack.pop_front().unwrap();
        let neighbor_locs = data[node_index as usize].borrow_mut().to_owned().neighbors;
        for loc in neighbor_locs.iter(){
            let row = loc[0];
            let col = loc[1];

            let neighbor_id = id_hash(row, col, cols);

            //if the index is within the data set
            if let Some(element)  = data.get(neighbor_id as usize){
                
                let neighbor_is_explored =  element.borrow().is_explored;
                let neighbor_accum = element.borrow().accum;

                if !neighbor_is_explored {
                    if neighbor_accum >= threshold {
                        drainage.push(neighbor_id);
                    } else {
                        stack.push_back(neighbor_id);
                    }
                    
                    element.borrow_mut().is_explored = true;
                    explored.push(neighbor_id);
                }
                
            }

        }
    }

    //reset explored nodes
    for explored_index in explored.iter() {
        if let Some(element)  = data.get(explored_index.to_owned() as usize){
            element.borrow_mut().is_explored = false;
        }
    }

    return drainage;

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
    cols:u32,
    threshold: f64,
    path:&mut Vec<u32>,
    result:&mut Vec<Vec<u32>>,
    data: &Vec<RefCell<Node>>) {
        
    let mut path = path.to_owned();
    path.push(node_id);
    if node_id == end {
        result.push(path.clone());
    } else {
        let neighbor_locs = data[node_id as usize].borrow_mut().to_owned().neighbors;
        for loc in neighbor_locs.iter() {
            let row = loc[0];
            let col = loc[1];
            let neighbor_id = &id_hash(row, col, cols);

            let neighbor_accum = data[*neighbor_id as usize].borrow_mut().to_owned().accum;
            if (*neighbor_id != end) && (path.len() > 2) {
                //recursion in here
                if (neighbor_accum < threshold) && (!path.contains(neighbor_id)) {
                    dfs_modified(*neighbor_id,
                                end,
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
                                cols,
                                threshold,
                                &mut path,
                                result,
                                data);
                }
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
    cols:u32,
    threshold:f64,
    data: &Vec<RefCell<Node>>) -> Vec<Vec<u32>> {
    
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(100);
    let mut path: Vec<u32> = Vec::with_capacity(100);
    dfs_modified(start, end, cols, threshold, &mut path, &mut result, data);
    
    return result;
}

/// This function returns the id of the chosen drainage cell passing through the different
/// candidate paths.
/// 
/// Alpha controls the *path length bias* over the *accumulation bias* on the basis that
/// the shorter the path the better the candidate and the higher average accumulation the better
fn select_paths(paths:&Vec<Vec<u32>>,data:&Vec<Vec<Node>> ,alpha:f64) ->u32 {
    


    //placeholder
    0
}

fn main() {
    let file = File::open("./accumulations/cells_mini.json").expect("Failed to open file");

    println!("Loading graph data... (This may take a while)");
    let start = Instant::now();
    let deserialized: IndexMap<u32,Node> = serde_json::from_reader(file).expect("Failed to deserialize JSON file.");
    let vec_data:Vec<Node> = deserialized.values().cloned().collect();
    let mut data: Vec<RefCell<Node>> = vec_data
                .iter()
                .cloned()
                .map(|item|RefCell::new(item))
                .collect();
    

    let elapsed = start.elapsed();
    println!("JSON data successfuly serialized. Calculating HAND values ...");
    println!("Elapsed time: {:.2?}",elapsed);


    let rows: u32 = 5;//1047;
    let cols: u32 = 6;//1613;
    let data_len = (rows * cols) as usize;

    let mut hand: Array2<f64> = Array2::from_elem((rows as usize,cols as usize),-1.0);

    let pb = ProgressBar::new((rows * cols) as u64);
    let sty = ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-");
    pb.set_style(sty);

    let max_drainage:usize = 5;
    let mut closest_drainage:Vec<Drainage> = Vec::with_capacity(data_len);

    let start = Instant::now();
    //let mut num_processed:u128 = 0;
    let drainage_threshold:f64 = 5.0;
    for r in 0..rows {
        for c in 0..cols {
            
            let start_index = id_hash(r,c,cols);
            let mut drainage_ids = search_drainage(start_index, &mut data,drainage_threshold,cols);
            let distance_from_cell:Vec<u32> = drainage_ids
                                                .iter()
                                                .map(|this_id|{
                                                    manhattan(*this_id, start_index, cols)
                                                })
                                                .collect();
            drainage_ids.sort_by(|val_a,val_b| {
                let dist_a = manhattan(*val_a,start_index, cols);
                let dist_b  = manhattan(*val_b,start_index, cols);
                //least to greatest
                dist_a.cmp(&dist_b)
            });
            
            let drainage_len = drainage_ids.len();
            let num_to_store:usize = drainage_len.min(max_drainage);

            let drainage = Drainage{node_id: start_index,closest:drainage_ids[0..num_to_store].to_vec()};
            closest_drainage.push(drainage);

            hand[[r as usize, c as usize]] = drainage_len as f64;
            

            //num_processed += 1;
            pb.inc(1);
        }
    }
    
    pb.finish();
    let elapsed = start.elapsed();
    println!("Connected river cells identified.");
    println!("Elapsed time: {:.2?}",elapsed);

    let closest_drainage_rfc: Vec<RefCell<Drainage>> = closest_drainage
                    .iter()
                    .cloned()
                    .map(|drainage| RefCell::new(drainage))
                    .collect();

    //try searching possible paths
    let start_node:u32 = 5;
    let end_node = closest_drainage_rfc[start_node as usize].borrow_mut().closest[2];
    let all_paths = find_all_paths(start_node, end_node, cols, drainage_threshold, &data);
    println!("\nGetting all possible paths for:");
    println!("Start: {start_node}, End: {end_node}");
    for (i,path) in all_paths.iter().enumerate(){
        println!("{}. {:?}",i+1,*path);
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


}
