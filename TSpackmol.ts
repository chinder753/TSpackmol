import { Tensor, Tensor1D, Tensor2D, Rank } from "@tensorflow/tfjs";
import * as tf from '@tensorflow/tfjs'
// just for test
import * as fs from 'fs'



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



class Structure {
    protected natom:number;
    protected elements:string[];
    protected coordinates:Tensor2D;

    // overload
    constructor(xyz_file:string);  // input XYZ file
    constructor(elements:string[], coordinates:number[][]);   // input elements in a string array, and coordinates in n by 3 number array
    // define
    constructor(elements:string|string[], coordinates?:number[][]){
        if ((elements instanceof Array) && (coordinates instanceof Array)) { // input array
            this.parse_array(elements, coordinates)
        } else if ((typeof elements == 'string') && (typeof coordinates == 'undefined')) {  // input xyz file
            let error_line = this.parse_xyz_file(elements)
            if(error_line != -1) throw "something wrong in input file " + error_line.toString() + " lines";
        } else {
            throw "something wrong in input variable type: \n"
                + "  elements type: " + typeof elements 
                + "  coordinates type: " + typeof coordinates;
        }
    }

    // parse the XYZ file in string to Structure, success return -1, if not return the line number
    public parse_xyz_file(xyz_file:string):number {
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
        // array to Structure
        this.parse_array(temp_elements, temp_coordinates);
        return -1;
    }

    // trans the elements array and coordinates array to Structure
    public parse_array(elements:string[], coordinates:number[][]) {
        if(elements.length != coordinates.length) {
            throw "elements and coordinates length are not equal";
        }
        this.elements = elements;
        this.coordinates = tf.tidy(() => {
            let temp_coordinates = tf.tensor2d(coordinates, [coordinates.length, 3], 'float32');
            return tf.sub(temp_coordinates, temp_coordinates.mean(0));
        });
    }

    // remove the atom that index is n, the index starts from 0
    public remove_atom(n:number){
        this.natom -= 1;
        this.elements.splice(n, 1);
        this.coordinates = tf.tidy(() => {
            let temp_list = this.coordinates.unstack();
            this.coordinates.dispose();
            temp_list.splice(n, 1);
            return <Tensor2D>tf.stack(temp_list);
        });
    }

    // parse elements and coordinates to string in XYZ 
    public async to_XYZ(coordinates:Tensor2D = this.coordinates):Promise<{natom:number, xyz:string}> {
        let natom = this.natom;
        let xyz = '';
        return coordinates.array().then(
            (vale):{natom:number, xyz:string} => {
                this.elements.forEach((ele, index) => {
                    xyz += ele;
                    vale[index].forEach((x) => {
                        xyz += "    ";
                        xyz += String(x.toPrecision(12));
                    })
                    xyz += '\n';
                });
                coordinates.dispose();
                return {natom, xyz};
            },
            () => {
                throw "some wrong in Structure to_XYZ()";
            }
        );
    }

    public get_atoms(){
        return [this.natom, this.elements, this.coordinates];
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

}



class Molecule extends Structure{
    private angle:Tensor;
    private center:Tensor;
    private cell_vec:Tensor;

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
    public move(coordinates:Tensor) {
        this.center = tf.add(this.center, coordinates);
        this.coordinates = tf.add(this.coordinates, this.center);
    }

    // move the molecule to coordinates
    public move_to(coordinates:Tensor) {
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

    public get_cell(){
        return [this.angle, this.center, this.cell_vec];
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
    private mol_types:Structure[];
    private num_mol:number[];
    private mol:Structure[];
    private coordinates:Tensor2D;

    // creat a box
    constructor(box_size:number[], box_type:string){
        this.change_box(box_size, box_type);
        this.distance = 0;
        this.mol_types = [];
        this.num_mol = [];
        this.mol = [];
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
    public add_molecule(molecule:Structure, num:number){
        let index = this.mol_types.indexOf(molecule);
        if(index == -1){
            this.mol_types.push(molecule);
            this.num_mol.push(num);
        }else{
            this.num_mol[index] += num;
        }
    }

    public random_coordinates(seed:number = Date.now()){
        if(this.mol_types.length == 0) throw "there have no molecules in the box";
        var creat_rand_tensor;
        switch(this.box_type){
            case "cube":
                creat_rand_tensor = (val:number):Tensor2D => {return tf.randomUniform([val, 3], 0, this.box_size[0], 'float32', seed)};
                break;
            case "cuboid":
                break;
        }

        let nmol = 0;
        this.num_mol.forEach((num, index) => {
            let mol = this.mol_types[index];
            for(let i=0; i<num; i++){
                this.mol.push(mol);
            }
            nmol += num;
        });

        if(typeof this.coordinates != "undefined") this.coordinates.dispose();
        this.coordinates = creat_rand_tensor(nmol);
    }

    // calculate the distances for each atom in mol_1 and mol_2, if a distance is less than tolerance set it to 0
    // center of mol_1 is point M, mol_2 is N, atom_1 is A, atom_2 is B
    private async calc_distance_mol(index_1:number, center_1:Tensor1D, index_2:number, center_2:Tensor1D,tolerance:number):Promise<number> {
        let mol_1 = this.mol[index_1];
        let mol_2 = this.mol[index_2];

        let distance = 0;  // |AB|
        let R = center_2.sub(center_1);  // R: ON - OM = MN
        // loop in mol_1
        mol_1.get_coordinates().unstack().forEach((r_1) => {  // r_1: MA
            let zero = tf.zeros([mol_2.get_natom()]);
            distance += tf.tidy(() => {
                let delta_r = R.sub(r_1);  // delta_r: MN - MA = AN
                let each_r = delta_r.add(mol_2.get_coordinates());  // each_r: AN + NB = AB
                let each_d = tf.einsum("ij, ij -> i", each_r, each_r);
                let cond = tf.less(each_d, [tolerance]);
                return <number>tf.where(cond, each_d, zero).sqrt().sum().arraySync();
            });
        });
        return distance;
    }

    // calculate the distances for all atom in box
    public async calc_distance_box(tolerance:number, coordinates:Tensor2D = this.coordinates):Promise<number> {
        let array_center = <Tensor1D[]>coordinates.unstack();
        if(coordinates != this.coordinates) coordinates.dispose();

        let wait_XYZ:Promise<number>[] = [];
        for(let i=0; i<array_center.length; i++){
            for(let j=i+1; j<array_center.length; j++){
                wait_XYZ.push(this.calc_distance_mol(i, array_center[i], j, array_center[j], tolerance));
            }
        }

        let distance = await Promise.all(wait_XYZ).then((mdis) => {
            let temp = 0.0;
            mdis.forEach((val) => {
                temp += val;
            });
            return temp;
        });
        return distance;
    }

    public async to_XYZ(coordinates:Tensor2D = this.coordinates):Promise<{natom: number, xyz: string}> {
        let  array_center = await coordinates.array();
        if(array_center.length<=0) throw "molecules in box have no coordinate";

        let wait_XYZ:Promise<{natom: number, xyz: string}>[] = [];
        array_center.forEach((center, index) => {
            let mol = this.mol[index]
            wait_XYZ.push(mol.to_XYZ(<Tensor2D>mol.get_coordinates().add(center)));
        });
        if(coordinates != this.coordinates) coordinates.dispose();

        return Promise.all(wait_XYZ).then((val) => {
            let natom = 0;
            this.mol_types.forEach((type, index) => {
                natom += type.get_natom() * this.num_mol[index];
            });

            let xyz = "";
            val.forEach((val) => {
                xyz += val["xyz"];
            });

            return {natom, xyz};
        });
    }

    public get_molecules(){
        return this.mol;
    }
}

// test
let t_1 = Date.now();
var data = fs.readFileSync('H2O.xyz');
let h2o = new Structure(data.toString());
let box = new Packmol([50], "cube");
box.add_molecule(h2o, 1000);
box.random_coordinates();
let p = [box.to_XYZ(), box.calc_distance_box(2)];
Promise.all(p).then((val) => {
    console.log(String(val[0]["natom"])+"\ndistence error: "+ String(val[1]) +"\n"+val[0]["xyz"]+"\n")
    console.log(tf.memory(), "\n", String(val[1]), "\n", Date.now() - t_1);
});