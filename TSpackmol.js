"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
exports.__esModule = true;
var tf = require("@tensorflow/tfjs");
// just for test
var fs = require("fs");
var periodic_table = ['H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    'Uue'];
var Structure = /** @class */ (function () {
    // define
    function Structure(elements, coordinates) {
        if ((elements instanceof Array) && (coordinates instanceof Array)) { // input array
            this.prase_array(elements, coordinates);
        }
        else if ((typeof elements == 'string') && (typeof coordinates == 'undefined')) { // input xyz file
            var error_line = this.prase_xyz_file(elements);
            if (error_line != -1)
                throw "something wrong in input file " + error_line.toString() + " lines";
        }
        else {
            throw "something wrong in input variable type: \n"
                + "  elements type: " + typeof elements
                + "  coordinates type: " + typeof coordinates;
        }
    }
    // prase the XYZ file in string to Structure, success return -1, if not return the line number
    Structure.prototype.prase_xyz_file = function (xyz_file) {
        var contan_list = xyz_file.replace(/(\r*\n)+$/g, '').split(/\r*\n/); // remove excess \n at the end of the XYZ file, and split it
        this.natom = parseInt(contan_list[0]); // get the number of atoms
        if (isNaN(this.natom) || (contan_list.length < 3) || (this.natom + 2 != contan_list.length))
            return 1;
        // srting to array
        var temp_coordinates = [];
        var temp_elements = [];
        for (var i = 2; i < contan_list.length; i++) {
            var line = contan_list[i].replace(/^\s+|\s+$/g, '').split(/\s+/);
            if ((line.length != 4) || periodic_table.includes(line[0]) == false)
                return i + 1; // find the symbol read from XYZ file in periodic_table
            temp_elements.push(line[0]);
            temp_coordinates.push([parseFloat(line[1]), parseFloat(line[2]), parseFloat(line[3])]);
        }
        // array to Structure
        this.prase_array(temp_elements, temp_coordinates);
        return -1;
    };
    // trans the elements array and coordinates array to Structure
    Structure.prototype.prase_array = function (elements, coordinates) {
        if (elements.length != coordinates.length) {
            throw "elements and coordinates length are not equal";
        }
        this.elements = elements;
        this.coordinates = tf.tidy(function () {
            var temp_coordinates = tf.tensor2d(coordinates, [coordinates.length, 3], 'float32');
            return tf.sub(temp_coordinates, temp_coordinates.mean(0));
        });
    };
    // remove the atom that index is n, the index starts from 0
    Structure.prototype.remove_atom = function (n) {
        var _this = this;
        this.natom -= 1;
        this.elements.splice(n, 1);
        this.coordinates = tf.tidy(function () {
            var temp_list = _this.coordinates.unstack();
            _this.coordinates.dispose();
            temp_list.splice(n, 1);
            return tf.stack(temp_list);
        });
    };
    // prase elements and coordinates to string in XYZ 
    Structure.prototype.to_XYZ = function (coordinates) {
        if (coordinates === void 0) { coordinates = this.coordinates; }
        return __awaiter(this, void 0, void 0, function () {
            var natom, xyz;
            var _this = this;
            return __generator(this, function (_a) {
                natom = this.natom;
                xyz = '';
                return [2 /*return*/, coordinates.array().then(function (vale) {
                        _this.elements.forEach(function (ele, index) {
                            xyz += ele;
                            vale[index].forEach(function (x) {
                                xyz += "    ";
                                xyz += String(x.toPrecision(12));
                            });
                            xyz += '\n';
                        });
                        coordinates.dispose();
                        return { natom: natom, xyz: xyz };
                    }, function () {
                        throw "some wrong in Structure to_XYZ()";
                    })];
            });
        });
    };
    Structure.prototype.get_atoms = function () {
        return [this.natom, this.elements, this.coordinates];
    };
    Structure.prototype.get_natom = function () {
        return this.natom;
    };
    Structure.prototype.get_elements = function () {
        return this.elements;
    };
    Structure.prototype.get_coordinates = function () {
        return this.coordinates;
    };
    return Structure;
}());
var Molecule = /** @class */ (function (_super) {
    __extends(Molecule, _super);
    function Molecule() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    // calculate the geometry center of the molecule
    Molecule.prototype.calc_center = function () {
        this.center = tf.mean(this.coordinates, 0, false);
    };
    // calculate diagonal vector of cell
    Molecule.prototype.calc_cell = function () {
        this.cell_vec = tf.abs(tf.sub(tf.max(this.coordinates, 0), this.center));
    };
    // move the molecule to zero point
    Molecule.prototype.move_to_zero = function () {
        this.coordinates = tf.sub(this.coordinates, this.center);
        this.center = tf.zeros([3], 'float32');
    };
    // add coordinates to molecule coordinates
    Molecule.prototype.move = function (coordinates) {
        this.center = tf.add(this.center, coordinates);
        this.coordinates = tf.add(this.coordinates, this.center);
    };
    // move the molecule to coordinates
    Molecule.prototype.move_to = function (coordinates) {
        this.coordinates = tf.add(this.coordinates, tf.sub(coordinates, this.center));
        this.center = coordinates;
    };
    // rotate the molecule alpha radian about X-axis, beta radian about Y-axis, gamma radian about Z-axis
    Molecule.prototype.rotate = function (alpha, beta, gamma) {
        if (alpha === void 0) { alpha = 0; }
        if (beta === void 0) { beta = 0; }
        if (gamma === void 0) { gamma = 0; }
        this.angle = tf.add(this.angle, [alpha, beta, gamma]);
        this.coordinates = tf.sub(this.coordinates, this.center);
        if (alpha != 0)
            this.coordinates = tf.matMul(this.coordinates, [[1, 0, 0], [0, Math.cos(alpha), -Math.sin(alpha)], [0, Math.sin(alpha), Math.cos(alpha)]]);
        if (beta != 0)
            this.coordinates = tf.matMul(this.coordinates, [[Math.cos(beta), 0, Math.sin(beta)], [0, 1, 0], [-Math.sin(beta), 0, Math.cos(beta)]]);
        if (gamma != 0)
            this.coordinates = tf.matMul(this.coordinates, [[Math.cos(gamma), -Math.sin(gamma), 0], [Math.sin(gamma), Math.cos(gamma), 0], [0, 0, 1]]);
        this.coordinates = tf.add(this.coordinates, this.center);
    };
    // rotate the molecule about X-axis to alpha radian, about Y-axis to beta radian, about Z-axis to gamma radian
    Molecule.prototype.rotate_to = function (alpha, beta, gamma) {
        if (alpha === void 0) { alpha = 0; }
        if (beta === void 0) { beta = 0; }
        if (gamma === void 0) { gamma = 0; }
        this.rotate(alpha - this.angle[0], beta - this.angle[1], gamma - this.angle[2]);
    };
    Molecule.prototype.get_cell = function () {
        return [this.angle, this.center, this.cell_vec];
    };
    Molecule.prototype.get_angle = function () {
        return this.angle;
    };
    Molecule.prototype.get_center = function () {
        return this.center;
    };
    Molecule.prototype.get_cell_vec = function () {
        return this.cell_vec;
    };
    return Molecule;
}(Structure));
var Packmol = /** @class */ (function () {
    // creat a box
    function Packmol(box_size, box_type) {
        this.change_box(box_size, box_type);
        this.distance = 0;
        this.mol_types = [];
        this.num_mol = [];
        this.mol = [];
    }
    // change the box size and type
    Packmol.prototype.change_box = function (box_size, box_type) {
        this.box_type = box_type;
        switch (box_type) {
            case "cube":
                if (box_size.length != 1)
                    throw "box_size just needs 1 parameter if box_type is 'cube'";
                break;
            case "cuboid":
                if (box_size.length != 3)
                    throw "box_size just needs 3 parameters if box_type is 'cuboid'";
                break;
        }
        this.box_size = box_size;
    };
    // add moelecule to box
    Packmol.prototype.add_molecule = function (molecule, num) {
        var index = this.mol_types.indexOf(molecule);
        if (index == -1) {
            this.mol_types.push(molecule);
            this.num_mol.push(num);
        }
        else {
            this.num_mol[index] += num;
        }
    };
    Packmol.prototype.random_coordinates = function (seed) {
        var _this = this;
        if (seed === void 0) { seed = Date.now(); }
        if (this.mol_types.length == 0)
            throw "there have no molecules in the box";
        var creat_rand_tensor;
        switch (this.box_type) {
            case "cube":
                creat_rand_tensor = function (val) { return tf.randomUniform([val, 3], 0, _this.box_size[0], 'float32', seed); };
                break;
            case "cuboid":
                break;
        }
        var nmol = 0;
        this.num_mol.forEach(function (num, index) {
            var mol = _this.mol_types[index];
            for (var i = 0; i < num; i++) {
                _this.mol.push(mol);
            }
            nmol += num;
        });
        if (typeof this.coordinates != "undefined")
            this.coordinates.dispose();
        this.coordinates = creat_rand_tensor(nmol);
    };
    // calculate the distances for each atom in mol_1 and mol_2, if a distance is less than tolerance set it to 0
    // center of mol_1 is point M, mol_2 is N, atom_1 is A, atom_2 is B
    Packmol.prototype.calc_distance_mol = function (index_1, center_1, index_2, center_2, tolerance) {
        return __awaiter(this, void 0, void 0, function () {
            var mol_1, mol_2, distance, R;
            return __generator(this, function (_a) {
                mol_1 = this.mol[index_1];
                mol_2 = this.mol[index_2];
                distance = 0;
                R = center_2.sub(center_1);
                // loop in mol_1
                mol_1.get_coordinates().unstack().forEach(function (r_1) {
                    var zero = tf.zeros([mol_2.get_natom()]);
                    distance += tf.tidy(function () {
                        var delta_r = R.sub(r_1); // delta_r: MN - MA = AN
                        var each_r = delta_r.add(mol_2.get_coordinates()); // each_r: AN + NB = AB
                        var each_d = tf.einsum("ij, ij -> i", each_r, each_r);
                        var cond = tf.less(each_d, [tolerance]);
                        return tf.where(cond, each_d, zero).sqrt().sum().arraySync();
                    });
                });
                return [2 /*return*/, distance];
            });
        });
    };
    // calculate the distances for all atom in box
    Packmol.prototype.calc_distance_box = function (tolerance, coordinates) {
        if (coordinates === void 0) { coordinates = this.coordinates; }
        return __awaiter(this, void 0, void 0, function () {
            var array_center, wait_XYZ, i, j, distance;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        array_center = coordinates.unstack();
                        if (coordinates != this.coordinates)
                            coordinates.dispose();
                        wait_XYZ = [];
                        for (i = 0; i < array_center.length; i++) {
                            for (j = i + 1; j < array_center.length; j++) {
                                wait_XYZ.push(this.calc_distance_mol(i, array_center[i], j, array_center[j], tolerance));
                            }
                        }
                        return [4 /*yield*/, Promise.all(wait_XYZ).then(function (mdis) {
                                var temp = 0.0;
                                mdis.forEach(function (val) {
                                    temp += val;
                                });
                                return temp;
                            })];
                    case 1:
                        distance = _a.sent();
                        return [2 /*return*/, distance];
                }
            });
        });
    };
    Packmol.prototype.to_XYZ = function (coordinates) {
        if (coordinates === void 0) { coordinates = this.coordinates; }
        return __awaiter(this, void 0, void 0, function () {
            var array_center, wait_XYZ;
            var _this = this;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, coordinates.array()];
                    case 1:
                        array_center = _a.sent();
                        if (array_center.length <= 0)
                            throw "molecules in box have no coordinate";
                        wait_XYZ = [];
                        array_center.forEach(function (center, index) {
                            var mol = _this.mol[index];
                            wait_XYZ.push(mol.to_XYZ(mol.get_coordinates().add(center)));
                        });
                        if (coordinates != this.coordinates)
                            coordinates.dispose();
                        return [2 /*return*/, Promise.all(wait_XYZ).then(function (val) {
                                var natom = 0;
                                _this.mol_types.forEach(function (type, index) {
                                    natom += type.get_natom() * _this.num_mol[index];
                                });
                                var xyz = "";
                                val.forEach(function (val) {
                                    xyz += val["xyz"];
                                });
                                return { natom: natom, xyz: xyz };
                            })];
                }
            });
        });
    };
    Packmol.prototype.get_molecules = function () {
        return this.mol;
    };
    return Packmol;
}());
// test
var t_1 = Date.now();
var data = fs.readFileSync('H2O.xyz');
var h2o = new Structure(data.toString());
var box = new Packmol([50], "cube");
box.add_molecule(h2o, 1000);
box.random_coordinates();
var p = [box.to_XYZ(), box.calc_distance_box(2)];
Promise.all(p).then(function (val) {
    console.log(String(val[0]["natom"]) + "\ndistence error: " + String(val[1]) + "\n" + val[0]["xyz"] + "\n");
    console.log(tf.memory(), "\n", String(val[1]), "\n", Date.now() - t_1);
});
