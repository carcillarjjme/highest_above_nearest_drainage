use std::borrow::BorrowMut;
use std::fs::File;
use std::io::Read;
use std::collections::{HashMap,VecDeque};
use std::process::id;
use serde::{Deserialize, Serialize};
use std::cell::{RefCell,RefMut};
use ndarray::Array2;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyExt};
use std::env;
use std::io::BufWriter;
use std::time::Instant;
use indexmap::IndexMap;


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



fn id_hash(row:u32,col:u32,cols:u32) -> u32 {
    return row*cols + col%cols;
}

fn reverse_hash(node_id:u32,cols:u32) -> (u32,u32){
    let row = node_id/cols; //integer division
    let col = node_id - row*cols;
    return (row,col)
}

fn same_node(node_a:Node, node_b:Node) -> bool {
    if (node_a.row == node_b.row) && (node_a.col == node_b.col) {
        return true
    }
    return false;
}

fn search_drainage(start_index:u32,data:&mut Vec<RefCell<Node>>,threshold:f64,cols:u32) ->Vec<u32> {
    
    let mut explored:Vec<u32> = Vec::new();
    let mut stack:VecDeque<u32> = VecDeque::new();
    let mut drainage:Vec<u32> = Vec::new();
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


fn main() {
    let file = File::open("./accumulations/cells.json").expect("Failed to open file");

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

    let rows: u32 = 1047;
    let cols: u32 = 1613;

    let mut hand: Array2<f64> = Array2::from_elem((rows as usize,cols as usize),-1.0);
    
    let start = Instant::now();
    let mut num_processed:u128 = 0;
    let drainage_threshold = 1000.0;
    for r in 0..rows {
        for c in 0..cols {
            if num_processed % 10000 == 0 {
                let progress = (num_processed * 100) /(rows as u128*cols as u128);
                println!("[Nearby Paths] Progress {progress:.2}%");
            }
            let start_index = id_hash(r,c,cols);
            let drainage = search_drainage(start_index, &mut data,drainage_threshold,cols);
            hand[[r as usize, c as usize]] = drainage.len() as f64;
            num_processed += 1;
        }
    }
    let elapsed = start.elapsed();
    println!("Connected river cells identified.");
    println!("Elapsed time: {:.2?}",elapsed);

    /*let coords: Vec<(u32,u32,f64)>= drainage
                        .iter()
                        .map(
                            |item| {
                                let node = data[*item as usize].borrow();
                                (node.row,node.col,node.accum)
                            })
                        .collect();

    println!("Start: {:?}, Drainage Count: {},\nDrainage Nodes:\n{:?}",reverse_hash(start_index, cols),drainage.len(),drainage);
    for (i,coord) in coords.iter().enumerate() {

        println!("{}. {:?}",i+1,coord);
    }*/

    
    let writer = BufWriter::new(File::create("./accumulations/hand.npy").unwrap());
    match hand.write_npy(writer){
        Ok(_) => {println!("{} was written.","./accumulations/hand.npy")},
        Err(err) => {println!("An error occured while writing {}. See below:\n{}","./accumulations/hand.npy",err)},
    };

}
