import { Tensor } from "@tensorflow/tfjs";
import { Tensor1D, Tensor2D} from "@tensorflow/tfjs-core";


const tf = require('@tensorflow/tfjs');
const _ = require('lodash');
const fs = require("fs");


const periodic_table = [ 'H','He'
                        ,'Li','Be','B','C','N','O','F','Ne'
                        ,'Na','Mg','Al','Si','P','S','Cl','Ar'
                        ,'K','Ca'
                        ,'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn'
                        ,'Ga','Ge','As','Se','Br','Kr'
                        ,'Rb','Sr'
                        ,'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd'
                        ,'In','Sn','Sb','Te','I','Xe'
                        ,'Cs','Ba'
                        ,'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'
                        ,'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg'
                        ,'Tl','Pb','Bi','Po','At','Rn'
                        ,'Fr','Ra'
                        ,'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'
                        ,'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn'
                        ,'Nh','Fl','Mc','Lv','Ts','Og'
                        ,'Uue'];



class Molecule {
    private angle:Tensor1D;
    private center:Tensor1D;
    private cell_vec:Tensor1D;
    private natom:number;
    private elements:string[];
    private coordinates:Tensor2D;

    // overload
    constructor(xyz_file:string);  // input XYZ file
    constructor(elements:string[], coordinates:number[][]);   // input elements in a string array, and coordinates in n by 3 number array
    // define
    constructor(elements:string|string[], coordinates?:number[][]){
        if ((elements instanceof Array) && (coordinates instanceof Array)) { // input array
            this.prase_array(elements, coordinates)
        } else if ((typeof elements == 'string') && (typeof coordinates == 'undefined')) {  // input xyz file
            let error_line = this.prase_xyz_file(elements)
            if(error_line != -1) throw "something wrong in input file " + error_line.toString() + " lines";
        } else {
            throw "something wrong in input variable type: \n"
                + "  elements type: " + typeof elements 
                + "  coordinates type: " + typeof coordinates;
        }
    }

    // prase the XYZ file in string to Molecule, success return -1, if not return the line number
    public prase_xyz_file(xyz_file:string):number {
        let contan_list:string[] = xyz_file.replace(/(\r*\n)+$/g,'').split(/\r*\n/);  // remove excess \n at the end of the XYZ file, and split it
        this.natom = parseInt(contan_list[0]);  // get the number of atoms
        if(isNaN(this.natom) || (contan_list.length<3) || (this.natom+2 != contan_list.length)) return 1;
        // srting to array
        var temp_coordinates:number[][] = [];
        var temp_elements:string[] = [];
        for(let i=2; i<contan_list.length; i++){
            let line:string[] = contan_list[i].replace(/^\s+|\s+$/g, '').split(/\s+/);
            if((line.length != 4) || periodic_table.includes(line[0]) == false) return i+1;// find the symbol read from XYZ file in periodic_table
            temp_elements.push(line[0]);
            temp_coordinates.push([parseFloat(line[1]), parseFloat(line[2]), parseFloat(line[3])]);
        }
        // array to Molecule
        this.prase_array(temp_elements, temp_coordinates);
        return -1;
    }

    // trans the elements array and coordinates array to Molecule
    public prase_array(elements:string[], coordinates:number[][]) {
        if(elements.length != coordinates.length) {
            throw "elements and coordinates length are not equal";
        }
        this.elements = elements;
        this.coordinates = tf.tensor2d(coordinates, [coordinates.length, 3], 'float32');
        this.angle = tf.zeros([3]);
        this.calc_center();
        this.calc_cell();
    }

    // calculate the geometry center of the molecule
    private calc_center() {
        this.center = tf.mean(this.coordinates, 0, false);
    }

    // calculate diagonal vector of cell
    private calc_cell() {
        this.cell_vec = tf.abs(tf.sub(tf.max(this.coordinates, 0), this.center));
    }

    // move the molecule to zero point
    public move_to_zero(){
        this.coordinates = tf.sub(this.coordinates, this.center);
        this.center = tf.zeros([3], 'float32');
    }

    // add coordinates to molecule coordinates
    public move(coordinates:Tensor1D) {
        this.center = tf.add(this.center, coordinates);
        this.coordinates = tf.add(this.coordinates, this.center);
    }

    // move the molecule to coordinates
    public move_to(coordinates:Tensor1D) {
        this.coordinates = tf.add(this.coordinates, tf.sub(coordinates, this.center));
        this.center = coordinates;
    }

    // rotate the molecule alpha radian about X-axis, beta radian about Y-axis, gamma radian about Z-axis
    public rotate(alpha:number = 0, beta:number = 0, gamma:number = 0){
        this.angle = tf.add(this.angle, [alpha, beta, gamma]);
        this.coordinates = tf.sub(this.coordinates, this.center);
        if(alpha != 0) this.coordinates = tf.matMul(this.coordinates, [[1, 0, 0], [0, Math.cos(alpha), -Math.sin(alpha)], [0, Math.sin(alpha), Math.cos(alpha)]]);
        if(beta != 0) this.coordinates = tf.matMul(this.coordinates, [[Math.cos(beta), 0, Math.sin(beta)], [0, 1, 0], [-Math.sin(beta), 0, Math.cos(beta)]]);
        if(gamma != 0) this.coordinates = tf.matMul(this.coordinates, [[Math.cos(gamma), -Math.sin(gamma), 0], [Math.sin(gamma), Math.cos(gamma), 0], [0, 0, 1]]);
        this.coordinates = tf.add(this.coordinates, this.center);
    }

    // rotate the molecule about X-axis to alpha radian, about Y-axis to beta radian, about Z-axis to gamma radian
    public rotate_to(alpha:number = 0, beta:number = 0, gamma:number = 0){
        this.rotate(alpha-this.angle[0], beta-this.angle[1], gamma-this.angle[2]);
    }

    // remove the atom that index is n, the index starts from 0
    public remove_atom(n:number){
        this.natom -= 1;
        this.elements = this.elements.splice(n, 1);
        this.coordinates = tf.stack(tf.splice(this.coordinates, [0, 0], [n, 3]), tf.splice(this.coordinates, [n+1, 0], [this.natom-n, 3]) );
    }

    public to_XYZ():{natom:number, xyz:string} {
        let natom = this.natom;
        let xyz = '';
        let coordinates = this.coordinates.arraySync();
        this.elements.forEach((ele, index) => {
            xyz += ele;
            coordinates[index].forEach((x) => {
                xyz += "    ";
                xyz += String(x.toPrecision(12));
            })
            xyz += '\n';
        });
        
        return {natom, xyz};
    }

    public get_atoms(){
        return [this.natom, this.elements, this.coordinates];
    }

    public get_cell(){
        return [this.angle, this.center, this.cell_vec];
    }

    public get_natom(){
        return this.natom;
    }

    public get_elements(){
        return this.elements;
    }

    public get_coordinates(){
        return this.coordinates;
    }

    public get_angle(){
        return this.angle;
    }

    public get_center(){
        return this.center;
    }

    public get_cell_vec(){
        return this.cell_vec;
    }

}


class Packmol{
    private distance:number;
    private box_size:number[];
    private box_type:string;
    private molecule_types:Molecule[];
    private num_mol:number[];
    private molecules:Molecule[][];

    // creat a box
    constructor(box_size:number[], box_type:string){
        this.change_box(box_size, box_type);
        this.distance = 0;
        this.molecule_types = [];
        this.num_mol = [];
        this.molecules = [];
    }

    // change the box size and type
    public change_box(box_size:number[], box_type:string){
        this.box_type = box_type;
        switch(box_type){
            case "cube":
                if(box_size.length != 1) throw "box_size just needs 1 parameter if box_type is 'cube'";
                break;
            case "cuboid":
                if(box_size.length != 3) throw "box_size just needs 3 parameters if box_type is 'cuboid'";
                break;
        }
        this.box_size = box_size;
    }

    // add moelecule to box
    public add_molecule(molecule:Molecule, num:number){
        let index = this.molecule_types.indexOf(molecule);
        if(index == -1){
            this.molecule_types.push(molecule);
            this.num_mol.push(num);
        }else{
            this.num_mol[index] += num;
        }
    }

    public random_coordinates(seed:number = Date.now()){
        if(this.molecule_types.length == 0) throw "there have no molecules in the box";
        switch(this.box_type){
            case "cube":
                var creat_rand_tensor = (val:number):Tensor[] => {return tf.randomUniform([val, 3], 0, this.box_size[0], 'float32', seed).unstack()};
                break;
            case "cuboid":
                break;
        }
        this.num_mol.forEach((num, index) => {
            let mol = this.molecule_types[index];
            let temp_mol:Molecule[] = [];
            let temp_coordinates = creat_rand_tensor(num);
            for(let i=0; i<num; i++){
                mol.move_to(temp_coordinates[i].reshape([3]));
                temp_mol.push(_.clone(mol));
            }
            this.molecules.push(temp_mol);
        });
    }

    // calculate the distances for each atom in mol_1 and mol_2, if a distance is less than tolerance set it to 0
    // mol_x = [molecule type index, molecule coordinates index]
    private calc_distance_mol(index_1:number[], index_2:number[], tolerance:number):number {
        let mol_1 = this.molecules[index_1[0]][index_1[1]];
        let mol_2 = this.molecules[index_2[0]][index_2[1]];

        let distance = 0;
        let zero = tf.zeros([mol_2.get_natom()]);
        // loop in mol_1
        mol_1.get_coordinates().unstack().forEach((val) => {
            let delta_vec = tf.sub(val, mol_2.get_coordinates());
            // distance between val and another atom in mol_2 subtract tolerance
            let each_distance = tf.sub(tf.einsum('ia, ia -> i', delta_vec, delta_vec), [tolerance]);
            let cond = tf.less(each_distance, zero);
            // find the distance less then tolerance
            each_distance = tf.where(cond, each_distance, zero);
            distance += each_distance.sum(0).arraySync();
        });
        return distance;
    }

    // calculate the distances for all atom in box
    public calc_distance_box(tolerance:number):number {
        let distance = 0.0;
        let n_type = this.molecule_types.length;
        for(let type_1=0; type_1<n_type; type_1++){
            // calculate the distance between the same type of molecule
            for(let mol_1=0; mol_1<this.molecules[type_1].length; mol_1++){
                for(let mol_2=mol_1+1; mol_2<this.molecules[type_1].length; mol_2++){
                    distance += this.calc_distance_mol([type_1, mol_1], [type_1, mol_2], tolerance);
                }
            }
            // calculate the distance between different type of molecule
            for(let type_2=type_1+1; type_2<n_type; type_2++){
                for(let mol_1=0; mol_1<this.molecules[type_1].length; mol_1++){
                    for(let mol_2=0; mol_2<this.molecules[type_2].length; mol_2++){
                        distance += this.calc_distance_mol([type_1, mol_1], [type_2, mol_2], tolerance);
                    }
                }
            }
        }
        return distance;
    }

    public to_XYZ():{natom:number, xyz:string} {
        let natom = 0;
        this.molecule_types.forEach((type, index) => {
            natom += type.get_natom()*this.num_mol[index];
        });

        let xyz = "";
        this.molecules.forEach((type) => {
            type.forEach((mol) => {
                xyz += mol.to_XYZ()["xyz"];
            });
        });

        return {natom, xyz};
    }

    public get_molecules(){
        return this.molecules;
    }
}

// test
var data = fs.readFileSync('H2O.xyz');
let h2o = new Molecule(data.toString());
h2o.move_to_zero();
let box = new Packmol([10], "cube");
box.add_molecule(h2o, 10);
box.random_coordinates();
let temp_xyz = box.to_XYZ();
console.log(String(temp_xyz["natom"])+"\ndistence error: "+ String(box.calc_distance_box(2)) +"\n"+temp_xyz["xyz"]+"\n")