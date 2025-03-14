

# region pre_process

# endregion


# region load_data

# endregion


# region axon_growth_model
class generate_axon:
    def __init__(self):
        pass

    def find_existed_result(self):
        pass


class generate_axon_2D(generate_axon):
    def __init__(self, animal, atlas, Lam, beta, na, model_type, session, task_name, force_center_type, direction, length, grid_num, random_num, node_num, distribution_type, use_density_to_generate, try_load_previous_data=True):
        self.rearranged = False
        self.Lam = Lam
        self.beta = beta
        self.na = na
        self.model_type = model_type
        self.already_run = False
        self.session = session
        self.force_center_type = force_center_type
        self.direction = direction
        self.length = length
        self.animal = animal
        self.grid_num = grid_num
        self.random_num = random_num
        self.node_num = node_num
        self.distribution_type = distribution_type
        self.target_name, self.source_name, self.full_name, self.coords_target_name, self.all_name_coords, self.accurate_name_dict, self.name2namelist_dict = get_names(self.animal)
        self.use_density_to_generate = use_density_to_generate
        self.pruning_by_density = pruning_by_density
        self.pruning_by_hierarchy = pruning_by_hierarchy
        self.pruning_by_category = pruning_by_category
        # whether we use density to generate neuron's or not, we load density data first.
        self.density, self.density_exist, self.max_density, self.min_density = load_density(self.animal)
        # if use_density_to_generate is True, density_to_generate should be the same as data density(if data density exists)
        self.density_to_generate = {}
        if self.use_density_to_generate:
            if self.density_exist:
                print('----------------------------')
                print('use data density to generate')
                print('----------------------------')
                self.density_to_generate = self.density
            else:
                print('----------------------------')
                print('there is no density data, thus change use_density_to_generate to False')
                print('----------------------------')
                self.use_density_to_generate = False
                self.density_to_generate = self.density
        else:
            print('----------------------------')
            print('use uniform density to generate')
            print('----------------------------')
            for area in self.source_name:
                self.density_to_generate[area] = 1
        # note that if we dont have density data, pruning density should be changed to False
        if (self.density_exist==False) and (self.pruning_by_density==True):
            print('----------------------------')
            print('there is no density data, thus pruning by density is chaged to False')
            print('----------------------------')
            self.pruning_by_density = False
        # load hierarchy data
        self.hierarchy, self.hierarchy_exist = load_hierarchy(self.animal)
        # note that if we dont have hierarchy data, pruning hierarchy should be changed to False
        if (self.hierarchy_exist==False) and (self.pruning_by_hierarchy==True):
            print('----------------------------')
            print('there is no hierarchy data, thus pruning by hierarchy is chaged to False')
            print('----------------------------')
            self.pruning_by_hierarchy = False
        # load category data
        self.category = load_category(self.animal)
        self.category_exist = True
        # note that if we dont have category data, pruning category should be changed to False
        if (self.category_exist==False) and (self.pruning_by_category==True):
            print('----------------------------')
            print('there is no category data, thus pruning by category is chaged to False')
            print('----------------------------')
            self.pruning_by_category = False

        self.mesh_dict = {'macaque':77, 'marmoset':77, 'square_monkey':25, 'mouse':25, 'human':200}
        config = {'Lam': str(Lam), 'beta': str(beta), 'na': str(na), 'model_type': str(model_type), 'force_center_type': str(force_center_type), 'direction': str(direction), 'length': str(length), 'animal': str(animal), 'grid_num': str(grid_num), 'random_num': str(random_num),  'node_num': str(node_num), 'distribution_type': str(distribution_type), 'use_density_to_generate': str(self.use_density_to_generate), 'pruning_by_density': str(pruning_by_density), 'pruning_by_hierarchy': str(pruning_by_hierarchy), 'pruning_by_category': str(pruning_by_category)}
        with open(r'../../data/'+self.animal+'/name_location.pickle', 'rb') as file:
            self.name_location = pickle.load(file)
        mkdir(r'../../results/')
        mkdir(r'../../results/'+str(self.animal)+'/')
        mkdir(r'../../results/'+str(self.animal)+'/2D/')
        mkdir(r'../../results/'+str(self.animal)+'/2D/'+task_name+'/')
        mkdir(r'../../results/'+str(self.animal)+'/2D/'+task_name+'/'+model_type)
        mkdir(r'../../results/'+str(self.animal)+'/2D/'+task_name+'/'+model_type+'/'+force_center_type)
        self.basedir = r'../../results/'+str(self.animal)+'/2D/'+task_name+'/'+model_type+'/'+force_center_type+'/density_'+str(self.use_density_to_generate)
        mkdir(self.basedir)
        if try_load_previous_data:
            filenames=os.listdir(self.basedir)
            for file in filenames:
                with open(self.basedir+'/'+str(file)+'/config.pkl', 'rb') as f:
                    config_exist = pickle.load(f)
                if config_exist==config:
                    print('----------------------------')
                    print('this para config has already been calculated, the results are in' + self.basedir + '/' + file)
                    print('----------------------------')
                    self.already_run = True
                    self.basedir = self.basedir+'/'+file
                    break

        if self.already_run==False:
            # if haven't run this para config, then we need to generate a new folder, else, just use the existed folder
            now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            self.basedir = self.basedir+'/'+str(now)
            mkdir(self.basedir)
            with open(self.basedir+'/config.txt','w') as f:
                for key in config.keys():
                    f.write(str(key)+'='+str(config[key])+'\r\n')

            with open(self.basedir+'/config.pkl', 'wb') as f:
                pickle.dump(config, f)
            with open(self.basedir+'/density_to_generate.txt','w') as f:
                f.write(str(self.density_to_generate))
            mkdir(self.basedir+'/partial_data')
            mkdir(self.basedir+'/full_data')

        '''load data'''
        self.area_coords_whole, self.all_name_whole, self.pgon_whole, self.pts_whole = load_coords_data(Path('../../data/'+str(self.animal)+'/whole_cortex_coords/'))
        self.area_coords, self.all_name_coords, self.pgon, self.pts = load_coords_data(Path('../../data/'+str(self.animal)+'/area_coords/'))
        self.x_min = np.min(self.area_coords_whole['whole_cortex'].T[0])
        self.x_max = np.max(self.area_coords_whole['whole_cortex'].T[0])
        self.y_min = np.min(self.area_coords_whole['whole_cortex'].T[1])
        self.y_max = np.max(self.area_coords_whole['whole_cortex'].T[1])
        self.force_centers_dict = self.get_centers(self.force_center_type)
        self.N = len(self.full_name)
        print('--------directory---------')
        print(self.basedir)
        print('--------animal---------')
        print(self.animal)
        print('--------center_num---------')
        print(len(self.force_centers_dict.values()))
        print('--------targets_num---------')
        print(len(self.target_name))
        print('--------sources_num---------')
        print(len(self.source_name))
        
        self.distance_matrix_2D = pd.read_excel(r'../../data/'+animal+'/distance_matrix_2D.xlsx', index_col=0)
        self.distance_matrix_2D.index = self.distance_matrix_2D.index.astype(str)
        self.distance_matrix_2D.columns = self.distance_matrix_2D.columns.astype(str)

        self.distance_matrix_3D = pd.read_excel(r'../../data/'+animal+'/distance_matrix_3D.xlsx', index_col=0)
        self.distance_matrix_3D.index = self.distance_matrix_3D.index.astype(str)
        self.distance_matrix_3D.columns = self.distance_matrix_3D.columns.astype(str)


    def generate_axons(self):
        '''Generate axons using multiprocess and function force.'''
        print('----------generate axons----------')
        time_start = time.time()
        try:
            self.ss = np.load(self.basedir+'/full_data/ss_'+str(self.na)+'.npy')
            self.tt = np.load(self.basedir+'/full_data/tt_'+str(self.na)+'.npy')
            self.indices_ss = np.load(self.basedir+'/full_data/indices_ss_'+str(self.na)+'.npy')
            self.indices_tt = np.load(self.basedir+'/full_data/indices_tt_'+str(self.na)+'.npy')
            print('----------axons was generated, load them----------')
        except:
            process_list = []
            for part_index in range(self.session):
                p = Process(target=self.force, args=(self.na,self.beta,self.Lam,self.force_centers_dict,part_index,self.session,self.model_type))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
                
            # the code with multiprocess will not guarantee all of the process completed, so we need to try until success
            success = False
            while success==False:
                try:
                    success = True
                    self.ss = []
                    self.tt = []
                    self.indices_ss = []
                    self.indices_tt = []
                    for part_index in range(self.session):
                        if self.ss==[]:
                            self.ss = np.load(self.basedir+'/partial_data/ss_'+str(self.na)+'_'+str(part_index)+'.npy')
                            self.tt = np.load(self.basedir+'/partial_data/tt_'+str(self.na)+'_'+str(part_index)+'.npy')
                            self.indices_ss = np.load(self.basedir+'/partial_data/indices_ss_'+str(self.na)+'_'+str(part_index)+'.npy')
                            self.indices_tt = np.load(self.basedir+'/partial_data/indices_tt_'+str(self.na)+'_'+str(part_index)+'.npy')
                        else:
                            self.ss = np.concatenate([self.ss, np.load(self.basedir+'/partial_data/ss_'+str(self.na)+'_'+str(part_index)+'.npy')], axis=0)
                            self.tt = np.concatenate([self.tt, np.load(self.basedir+'/partial_data/tt_'+str(self.na)+'_'+str(part_index)+'.npy')], axis=0)
                            self.indices_ss = np.concatenate([self.indices_ss, np.load(self.basedir+'/partial_data/indices_ss_'+str(self.na)+'_'+str(part_index)+'.npy')], axis=0)
                            self.indices_tt = np.concatenate([self.indices_tt, np.load(self.basedir+'/partial_data/indices_tt_'+str(self.na)+'_'+str(part_index)+'.npy')], axis=0)
                    self.ss = np.reshape(self.ss, (-1, 2))
                    self.tt = np.reshape(self.tt, (-1, 2))
                    self.indices_ss = self.indices_ss.flatten()
                    self.indices_tt = self.indices_tt.flatten()
                    np.save(self.basedir+'/full_data/ss_'+str(self.na)+'.npy', self.ss)
                    np.save(self.basedir+'/full_data/tt_'+str(self.na)+'.npy', self.tt)
                    np.save(self.basedir+'/full_data/indices_ss_'+str(self.na)+'.npy', self.indices_ss)
                    np.save(self.basedir+'/full_data/indices_tt_'+str(self.na)+'.npy', self.indices_tt)
                except:
                    # if not completed then just wait for a while
                    time.sleep(10)
                    print('---------please wait---------')
                    # if the code enters the except block, success will be False
                    success = False
        time_end = time.time()
        time_c = time_end - time_start
        shutil.rmtree(self.basedir+'/partial_data/', ignore_errors=True)
        print('---------axons are generated---------')
        print('time cost', time_c, 's')


    def get_centers(self, center_type):
        '''Get centers for the calculation of force, center_type should be correspond to self.force_center_type.'''
        centers = {}
        if center_type=='boundary':
            for name in self.all_name_coords:
                cx,cy = np.mean(self.area_coords[name][:, 0]),np.mean(self.area_coords[name][:, 1])
                centers[name] = np.array([cx,cy])
        if center_type=='interior':
            for name in self.all_name_coords:
                cx,cy = np.mean(self.pts[name][:,0]),np.mean(self.pts[name][:,1])
                centers[name] = np.array([cx,cy])
        if center_type=='random':
            for i in range(self.random_num):
                centers[str(i)] = pointpats.random.poisson(self.pgon_whole['whole_cortex'],size=1)
        if center_type=='grid':
            x_list = np.linspace(self.x_min, self.x_max, int(self.grid_num), endpoint=True)
            y_list = np.linspace(self.y_min, self.y_max, int(self.grid_num), endpoint=True)
            i = 0
            for x in x_list:
                for y in y_list:
                    if self.whether_in_area([x, y], self.pgon_whole['whole_cortex'])==True:
                        centers[str(int(i))] = [x, y]
                        i = i + 1
        return centers


    def whether_in_area(self, p, poly):
        '''Judge whether the point p in the area occupied by poly.'''
        point = shapely.geometry.Point(p[0], p[1])
        return poly.intersects(point)


    def force_vector(self, points, beta, centers):
        """
        For each point (acctually s), using force_direction to get the target(t). The length between t and s should be 1.
        notation: s: initial positions of the axon
            t: terminal positions of the axon
            centers: positions of area centers
        """
        tt=[]
        if self.direction=='song':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    Ri = centers[center_index]
                    s = np.array(s)
                    forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                    forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='song_without_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if center_index!=self.polygon_index([s])[0]:
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='song_exclude_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if not self.whether_in_area(centers[center_index], self.pgon[self.polygon_index([s])[0]]):
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='exponential':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    Ri = centers[center_index]
                    s = np.array(s)
                    forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                    forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='exponential_without_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if center_index!=self.polygon_index([s])[0]:
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='muhan':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    Ri = centers[center_index]
                    s = np.array(s)
                    forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                    forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum
                direction_growth[0] = direction_growth[0]/np.sqrt(direction_growth[0]**2+direction_growth[1]**2)
                direction_growth[1] = direction_growth[1]/np.sqrt(direction_growth[0]**2+direction_growth[1]**2)
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        return tt


    def polygon_index(self, points):
        '''Get the area name for which the point belongs to.'''
        indices = []
        for s in points:
            bool_result = [self.whether_in_area(s, self.pgon[name]) for name in self.all_name_coords]
            try:
                idx = self.all_name_coords[bool_result.index(True)]
            except:
                idx = 'out_of_areas'
            indices.append(idx)
        return indices


    def force(self, na, beta, Lambda, centers, part_index, session, model_type='send'):
        """
        Using centers to generate na/session force, corresponding to beta, Lambda and model_type.
        Accelerated by multiprocess.
        notation: s: initial positions of the axon
                t: terminal positions of the axon
                centers: positions of N areas' center
                part_index: used as a random seed
                Lambda: scale parameter for the exponential sampling
                model_type: send means firstly generate source, receive means firstly generate target
        """
        ss=[]
        tt=[]
        indices_ss=[]
        indices_tt=[]
        # np.random.seed(part_index)
        np.random.seed(int(time.time()*10**6)%10**6)
        if self.length=='data_distribution_interior':
            CDF_node_length = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_length_interior.npy')
            node_length = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_length_interior.npy')
            def CDF_length_inverse(y):
                return np.interp(y, CDF_node_length, node_length)
        if self.length=='data_distribution_interior_area_specific':
            CDF_node_length = {}
            node_length = {}
            for source in self.source_name:
                CDF_node_length[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_'+source+'_length_interior.npy')
                node_length[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_'+source+'_length_interior.npy')
            def CDF_length_inverse(y, sourcename):
                accurate_sourcename = self.accurate_name_dict[sourcename]
                return np.interp(y, CDF_node_length[accurate_sourcename], node_length[accurate_sourcename])
        if self.length=='data_distribution_boundary':
            CDF_node_length = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_length_boundary.npy')
            node_length = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_length_boundary.npy')
            def CDF_length_inverse(y):
                return np.interp(y, CDF_node_length, node_length)
        if self.length=='data_distribution_boundary_area_specific':
            CDF_node_length = {}
            node_length = {}
            for source in self.source_name:
                CDF_node_length[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node'+source+'__length_boundary.npy')
                node_length[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node'+source+'__length_boundary.npy')
            def CDF_length_inverse(y, sourcename):
                accurate_sourcename = self.accurate_name_dict[sourcename]
                return np.interp(y, CDF_node_length[accurate_sourcename], node_length[accurate_sourcename])
        if self.direction=='data_distribution_interior':
            CDF_node_angle = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_angle_interior.npy')
            node_angle = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_angle_interior.npy')
            def CDF_angle_inverse(y):
                return np.interp(y, CDF_node_angle, node_angle)
        if self.direction=='data_distribution_interior_area_specific':
            CDF_node_angle = {}
            node_angle = {}
            for source in self.source_name:
                CDF_node_angle[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_'+source+'_angle_interior.npy')
                node_angle[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_'+source+'_angle_interior.npy')
            def CDF_angle_inverse(y, sourcename):
                accurate_sourcename = self.accurate_name_dict[sourcename]
                return np.interp(y, CDF_node_angle[accurate_sourcename], node_angle[accurate_sourcename])
        if self.direction=='data_distribution_boundary':
            CDF_node_angle = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_angle_boundary.npy')
            node_angle = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_angle_boundary.npy')
            def CDF_angle_inverse(y):
                return np.interp(y, CDF_node_angle, node_angle)
        if self.direction=='data_distribution_boundary_area_specific':
            CDF_node_angle = {}
            node_angle = {}
            for source in self.source_name:
                CDF_node_angle[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/CDF_node_'+source+'_angle_boundary.npy')
                node_angle[source] = np.load(r'./distribution/'+str(self.animal)+'/'+str(self.distribution_type)+'/'+self.model_type+'/'+str(self.node_num)+'/node_'+source+'_angle_boundary.npy')
            def CDF_angle_inverse(y, sourcename):
                accurate_sourcename = self.accurate_name_dict[sourcename]
                return np.interp(y, CDF_node_angle[accurate_sourcename], node_angle[accurate_sourcename])
            
        weighted_sum = 0
        for area in self.all_name_coords:
            weighted_sum = weighted_sum + self.density_to_generate[self.accurate_name_dict[area]]*self.pgon[area].area
        for area in self.all_name_coords:
            for i in range(int(na/session*self.density_to_generate[self.accurate_name_dict[area]]*self.pgon[area].area/weighted_sum)):
                s = pointpats.random.poisson(self.pgon[area],size=1)
                index_s = [area]
                # generate the t in the whole cortex
                generate_t = True
                try_num =0
                # to avoid endless iteration, if try_num is larger than 10, give up
                while generate_t==True and try_num<10:
                    try_num = try_num + 1
                    if self.direction=='song':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            Ri = centers[center_index]
                            forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                            forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                    
                    if self.direction=='song_without_self':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            if center_index!=index_s[0]:
                                Ri = centers[center_index]
                                forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                                forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon

                    if self.direction=='song_exclude_self':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            if not self.whether_in_area(centers[center_index], self.pgon[index_s[0]]):
                                Ri = centers[center_index]
                                forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                                forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon

                    if self.direction=='exponential':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            Ri = centers[center_index]
                            forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                            forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon

                    if self.direction=='exponential_without_self':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            if center_index!=index_s[0]:
                                Ri = centers[center_index]
                                forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                                forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon

                    if self.direction=='muhan':
                        forces=[]
                        # for center_index in self.all_name_coords:
                        for center_index in centers.keys():
                            Ri = centers[center_index]
                            forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                            forces.append(forcei)

                        if model_type=='send':
                            forces_sum = -np.sum(np.array(forces),axis=0)
                        else:
                            forces_sum = np.sum(np.array(forces),axis=0)
                        direction_growth = forces_sum
                        direction_growth[0] = direction_growth[0]/np.sqrt(direction_growth[0]**2+direction_growth[1]**2)
                        direction_growth[1] = direction_growth[1]/np.sqrt(direction_growth[0]**2+direction_growth[1]**2)

                    if (self.direction=='data_distribution_interior' or self.direction=='data_distribution_boundary'):
                        angle = CDF_angle_inverse(np.random.rand(1))
                        direction_growth = np.array([np.cos(angle), np.sin(angle)])
                        direction_growth = direction_growth.reshape((2,))
                        if model_type=='receive':
                            direction_growth = - direction_growth

                    if (self.direction=='data_distribution_interior_area_specific' or self.direction=='data_distribution_boundary_area_specific'):
                        if model_type=='receive':
                            print('area specific angle distribution cannot produce receive model direction')
                            print(1/0)#stop the code
                        else:
                            angle = CDF_angle_inverse(np.random.rand(1), index_s[0])
                            direction_growth = np.array([np.cos(angle), np.sin(angle)])
                            direction_growth = direction_growth.reshape((2,))

                    if self.direction=='random':
                        angle = np.random.rand(1) * 2 * np.pi
                        direction_growth = np.array([np.cos(angle), np.sin(angle)])
                        direction_growth = direction_growth.reshape((2,))
                        if model_type=='receive':
                            direction_growth = - direction_growth

                    if self.length=='exponential':
                        length = np.random.exponential(scale=Lambda[index_s[0]]) # length of the axon
                    if (self.length=='data_distribution_interior' or self.length=='data_distribution_boundary'):
                        length = CDF_length_inverse(np.random.rand(1))
                    if (self.length=='data_distribution_interior_area_specific' or self.length=='data_distribution_boundary_area_specific'):
                        if self.model_type=='receive':
                            print('area specific distance distribution cannot produce receive model length')
                            print(1/0)#stop the code
                        else:
                            length = CDF_length_inverse(np.random.rand(1), index_s[0])
                    if self.length=='power_law':
                        length = np.random.pareto(Lambda[index_s[0]])
                    t = s+length*direction_growth # terminal position of the axon
                    """
                    add constraints that axons must remain within the bounds of the cortex
                    """
                    if self.whether_in_area(t, self.pgon_whole['whole_cortex'])==True:
                        index_t = self.polygon_index([t])
                        if index_t==['out_of_areas']:
                            t_in_coords = False
                        else:
                            t_in_coords = True
                        generate_t = not t_in_coords
                        if t_in_coords==True:
                            """
                            if pruning_by_density is True, for a certain probability(related to the difference of density), reject this axon
                            """
                            if self.pruning_by_density:
                                reject = random.random()
                                if reject < (np.abs(self.density[self.accurate_name_dict[index_t[0]]]-self.density[self.accurate_name_dict[index_s[0]]])/(self.max_density-self.min_density)):
                                    # For large density difference, it should be less likely to exist a connection, thus it's more likely that reject<density difference, then we let generate_t=True to reject this axon.
                                    generate_t = True
                            """
                            if pruning_by_hierarchy is True, for a certain probability(related to the difference of hierarchy), reject this axon
                            """
                            if self.pruning_by_hierarchy:
                                reject = random.random()
                                try:
                                    if reject < np.abs(self.hierarchy[self.accurate_name_dict[index_t[0]]]-self.hierarchy[self.accurate_name_dict[index_s[0]]]):
                                        # For large hierarchy difference, it should be less likely to exist a connection, thus it's more likely that reject<hierarchy difference, then we let generate_t=True to reject this axon.
                                        generate_t = True
                                except:
                                    pass_try = 'pass'
                            """
                            if pruning_by_category is True, for a certain probability(related to the difference of category), reject this axon
                            """
                            if self.pruning_by_hierarchy:
                                reject = random.random()
                                if (reject<0.2) and (self.category[self.accurate_name_dict[index_t[0]]]!=self.category[self.accurate_name_dict[index_s[0]]]):
                                    # For different category, we reject the axon by center probability.
                                    generate_t = True

                    
                if generate_t==False:
                    # generate_t==False means t satisfies our condition
                    tt.append(t)
                    ss.append(s)
                    indices_tt.append(index_t)
                    indices_ss.append(index_s)


        if model_type=='receive':
            # just change ss and tt
            bb = ss
            ss = tt
            tt = bb
            indices_bb = indices_ss
            indices_ss = indices_tt
            indices_tt = indices_bb

        np.save(self.basedir+'/partial_data/ss_'+str(na)+'_'+str(part_index)+'.npy',ss)
        np.save(self.basedir+'/partial_data/tt_'+str(na)+'_'+str(part_index)+'.npy',tt)
        np.save(self.basedir+'/partial_data/indices_ss_'+str(na)+'_'+str(part_index)+'.npy',indices_ss)
        np.save(self.basedir+'/partial_data/indices_tt_'+str(na)+'_'+str(part_index)+'.npy',indices_tt)
        return ss, tt, indices_ss, indices_tt


    def FLN_model_and_data(self):
        '''Get the model FLN and Data FLN.'''
        '''Model FLN'''
        try:
            self.FLN_model = pd.read_excel(self.basedir+'/FLN_model.xlsx', index_col=0, header=0)
            self.FLN_model.index = self.FLN_model.index.astype(str)
            self.FLN_model.columns = self.FLN_model.columns.astype(str)

            self.connection_count_matrix = pd.read_excel(self.basedir+'/connection_count_matrix.xlsx', index_col=0, header=0)
            self.connection_count_matrix.index = self.connection_count_matrix.index.astype(str)
            self.connection_count_matrix.columns = self.connection_count_matrix.columns.astype(str)

            self.connection_count_matrix_with_self_connection = pd.read_excel(self.basedir+'/connection_count_matrix_with_self_connection.xlsx', index_col=0, header=0)
            self.connection_count_matrix_with_self_connection.index = self.connection_count_matrix_with_self_connection.index.astype(str)
            self.connection_count_matrix_with_self_connection.columns = self.connection_count_matrix_with_self_connection.columns.astype(str)
            print('--------model FLN is already calculated, load it--------')

        except:
            self.connection_count_matrix_with_self_connection = pd.DataFrame(np.zeros([self.N, self.N]), columns=self.full_name, index=self.full_name)
            for i in range(len(self.indices_ss)):
                accurate_source = self.accurate_name_dict[self.indices_ss[i]]
                accurate_target = self.accurate_name_dict[self.indices_tt[i]]
                self.connection_count_matrix_with_self_connection.loc[accurate_source, accurate_target] += 1
            '''because the data fln matrix doesn't have self connection, we remove it from connection_count_matrix'''
            self.connection_count_matrix = self.connection_count_matrix_with_self_connection.copy()
            for name in self.full_name:
                self.connection_count_matrix.loc[name, name] = 0
            all_connection_from = {}
            for target in self.full_name:
                all_connection_from[target] = 0
                for source in self.full_name:
                    all_connection_from[target] += self.connection_count_matrix.loc[source, target]
            FLN_model = pd.DataFrame(np.zeros([self.N, self.N]), columns=self.full_name, index=self.full_name)
            for source in self.full_name:
                for target in self.full_name:
                    FLN_model.loc[source, target] = self.connection_count_matrix.loc[source, target] / all_connection_from[target]
            # it may be some nan due to the division, replace it with 0
            FLN_model = FLN_model.fillna(0.0)
            self.FLN_model = FLN_model
            self.FLN_model.to_excel(self.basedir+'/FLN_model.xlsx')
            self.connection_count_matrix.to_excel(self.basedir+'/connection_count_matrix.xlsx')
            self.connection_count_matrix_with_self_connection.to_excel(self.basedir+'/connection_count_matrix_with_self_connection.xlsx')

        self.symmetry_CCM = pd.DataFrame((self.connection_count_matrix.values+self.connection_count_matrix.values.T)/2, index=self.connection_count_matrix.index, columns=self.connection_count_matrix.columns)
        self.symmetry_CCM.to_excel(self.basedir+'/DTI_model.xlsx')

        if self.animal in ['mouse', 'marmoset', 'macaque', 'marmoset-V4']:
            '''Data FLN'''
            self.FLN_data = pd.read_excel(r'../../data/'+str(self.animal)+'/FLN_data.xlsx', index_col=0)
            self.FLN_data.index = self.FLN_data.index.astype(str)
            self.FLN_data.columns = self.FLN_data.columns.astype(str)
            
        if self.animal in ['human', 'marmoset']:
            '''Data DTI'''
            self.DTI_data = pd.read_excel(r'../../data/'+str(self.animal)+'/DTI_data.xlsx', index_col=0)


    def mean_axon_length(self):
        '''Caculate mean axon length'''
        centers_interior_dict = self.get_centers('interior')
        all_neuron_num = np.sum(self.connection_count_matrix.values)
        center_length_sum = 0
        for target in self.connection_count_matrix.index:
            target_list = self.name2namelist_dict[target]
            for source in self.connection_count_matrix.columns:
                source_list = self.name2namelist_dict[source]
                distance = []
                for t in target_list:
                    for s in source_list:
                        distance.append(np.linalg.norm(centers_interior_dict[t]-centers_interior_dict[s]))
                center_length_sum = center_length_sum + self.connection_count_matrix.loc[source, target]*np.mean(distance)
        mean_center_length = center_length_sum / all_neuron_num
        axon_lenth_sum = 0
        for i in range(len(self.indices_ss)):
            if self.indices_ss[i]!=self.indices_tt[i]:
                axon_lenth_sum = axon_lenth_sum + np.linalg.norm(self.ss[i]-self.tt[i])
        mean_axon_length = axon_lenth_sum / all_neuron_num
        with open(self.basedir+'/distance.txt', 'w') as f:
            f.write('mean_center_length='+str(mean_center_length)+'\r\n')
            f.write('mean_axon_length='+str(mean_axon_length)+'\r\n')
        return mean_center_length, mean_axon_length


    def part_visualization(self, FLN_list, FLN_titles, file_name, plot_names, sources, targets, plot_type, draw_axon_arrow):
        
        print('donot use it')
        print(1/0)
        
        '''Draw several plots, green means the target location, red means source location. More red, larger FLN. FLN should be a list, for example, FLN_list = [FLN_data, FLN_model], FLN_titles should be the corresponding titles, for example, FLN_title = ['FLN_data', 'FLN_model']. Due to the incompleteness of data, we mask missing areas by black(this is only done when plot_type is source, because for plot_type=target, the data is complete.)'''
        color_dict = {'target': 'g', 'source': 'r'}
        target_source_transform = {'target': 'source', 'source': 'target'}
        arrow_size_dict = {'marmoset':0.25, 'macaque40':0.4, 'macaque29':0.4}
        for plot_name in plot_names:
            plot_name_list = self.name2namelist_dict[plot_name]
            figname = '_'+plot_type+'='+str(plot_name)+'.png'
            if plot_type=='source':
                figure ,axes = plt.subplots(1, len(FLN_list), figsize=(len(FLN_list)*10, 10))
                for iii in range(len(FLN_list)):
                    max_area = ''
                    FLN = FLN_list[iii]
                    axes[iii].plot(0, 0, label='target:green/source:red')
                    axes[iii].legend(fontsize=15)
                    for name in self.all_name_coords:
                        accurate_name = self.accurate_name_dict[name]
                        if plot_type=='source':
                            try:
                                alpha = FLN.loc[plot_name, accurate_name]/np.max(FLN.loc[plot_name, self.target_name])
                            except:
                                alpha = 0
                            if str(alpha)=='nan':
                                alpha = 0
                            elif np.allclose(alpha, 1):
                                max_area = accurate_name
                            if accurate_name not in self.target_name:
                                if accurate_name!=plot_name:
                                    axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c='black', zorder=1)
                            else:
                                axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], alpha=alpha, c=color_dict[target_source_transform[plot_type]], zorder=1)
                        boundary = self.area_coords[name]
                        boundary = np.append(boundary, [boundary[0]], axis=0)
                        axes[iii].plot(boundary.T[0], boundary.T[1], c='black', linewidth=1, zorder=1)
                        axes[iii].text(self.name_location[name][0], self.name_location[name][1], name, fontsize=7, font={'style':'normal','weight':'black'})
                    for name in plot_name_list:
                        axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c=color_dict[plot_type], zorder=1)
                    if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
                        # draw arrows
                        for i in range(len(sources)):
                            axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=arrow_size_dict[self.animal]/2, head_length=arrow_size_dict[self.animal]/2, zorder=2)
                            # if plot_type=='source':
                            #     for name in plot_name_list:
                            #         if self.whether_in_area([sources[i][0], sources[i][1]], self.area_coords[name]):
                            #             axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/4, (targets[i][1]-sources[i][1])/4, fc='b', ec='b', head_width=0.15, head_length=0.15)
                            # else:
                            #     axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=0.2, head_length=0.2)
                    if plot_type=='source':
                        axes[iii].set_title(FLN_titles[iii]+'\nmax fln in this plot: area '+max_area+'\n'+str(np.max(FLN.loc[plot_name, self.target_name])), fontdict={'weight': 'normal', 'size': 25})
                        if FLN_titles[iii]=='FLN_model' and draw_axon_arrow==True:
                            coords_plot_names = self.name2namelist_dict[plot_name]
                            for coords_plot_name in coords_plot_names:
                                index = np.where(((self.indices_ss.flatten()==coords_plot_name)*(self.indices_tt!=coords_plot_name))==True)
                                ss_partial = np.array(self.ss[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
                                tt_partial = np.array(self.tt[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
                                for i in range(len(ss_partial)):
                                    axes[iii].arrow(ss_partial[i][0], ss_partial[i][1], tt_partial[i][0]-ss_partial[i][0], tt_partial[i][1]-ss_partial[i][1], fc='orange', ec='orange', head_width=arrow_size_dict[self.animal], head_length=arrow_size_dict[self.animal], zorder=3)
                plt.savefig(str(file_name)+figname, dpi=300)
                plt.close()
            else:
                figure ,axes = plt.subplots(2, len(FLN_list), figsize=(len(FLN_list)*10, 2 * 10))
                for mask_index in range(2):
                    for iii in range(len(FLN_list)):
                        max_area = ''
                        FLN = FLN_list[iii]
                        axes[mask_index, iii].plot(0, 0, label='target:green/source:red')
                        axes[mask_index, iii].legend(fontsize=15)
                        for name in self.all_name_coords:
                            accurate_name = self.accurate_name_dict[name]
                            if plot_type=='target':
                                alpha = FLN.loc[accurate_name, plot_name]/np.max(FLN.loc[:, plot_name])
                                if str(alpha)=='nan':
                                    alpha = 0
                                elif np.allclose(alpha, 1):
                                    max_area = accurate_name
                                axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], alpha=alpha, c=color_dict[target_source_transform[plot_type]], zorder=1)
                            boundary = self.area_coords[name]
                            boundary = np.append(boundary, [boundary[0]], axis=0)
                            axes[mask_index, iii].plot(boundary.T[0], boundary.T[1], c='black', linewidth=1, zorder=1)
                            axes[mask_index, iii].text(self.name_location[name][0], self.name_location[name][1], name, fontsize=7, font={'style':'normal','weight':'black'})
                        for name in plot_name_list:
                            axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c=color_dict[plot_type], zorder=1)
                        if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
                            # draw arrows
                            for i in range(len(sources)):
                                axes[mask_index, iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=arrow_size_dict[self.animal]/2, head_length=arrow_size_dict[self.animal]/2, zorder=2)
                        if plot_type=='target':
                            if mask_index==0:
                                axes[mask_index, iii].set_title(FLN_titles[iii]+'\nmax fln in this plot: area '+max_area+'\n'+str(np.max(FLN.loc[:, plot_name])), fontdict={'weight': 'normal', 'size': 25})
                            if FLN_titles[iii]=='FLN_model' and draw_axon_arrow==True:
                                coords_plot_names = self.name2namelist_dict[plot_name]
                                for coords_plot_name in coords_plot_names:
                                    index = np.where(((self.indices_tt.flatten()==coords_plot_name)*(self.indices_ss!=coords_plot_name))==True)
                                    random.shuffle(index)
                                    ss_partial = np.array(self.ss[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
                                    tt_partial = np.array(self.tt[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
                                    for i in range(len(ss_partial)):
                                        axes[mask_index, iii].arrow(ss_partial[i][0], ss_partial[i][1], tt_partial[i][0]-ss_partial[i][0], tt_partial[i][1]-ss_partial[i][1], fc='orange', ec='orange', head_width=arrow_size_dict[self.animal], head_length=arrow_size_dict[self.animal], zorder=3)
                        if mask_index==1:
                            for name in self.all_name_coords:
                                accurate_name = self.accurate_name_dict[name]
                                if (accurate_name not in self.target_name) and (accurate_name!=plot_name):
                                    axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c='black', zorder=1)
                plt.savefig(str(file_name)+figname, dpi=300)
                plt.close()


    def simultaneously_visualization(self, FLN_list, FLN_titles, file_name, draw_axon_arrow, multiprocess):
        
        print('donot use it')
        print(1/0)
        
        '''Using part_visualization to generate FLN visualization plots, accelerated by multiprocess.'''
        if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
            # generate the vector field
            x_list = np.linspace(self.x_min, self.x_max, self.mesh_dict[self.animal])
            y_list = np.linspace(self.y_min, self.y_max, self.mesh_dict[self.animal])
            sources = []
            for x in x_list:
                for y in y_list:
                    if self.whether_in_area([x, y], self.pgon_whole['whole_cortex'])==True and self.polygon_index([[x, y]])[0]!='out_of_areas':
                        sources.append([x, y])
            targets = self.force_vector(sources,self.beta,self.force_centers_dict)
            sources = np.array(sources)
            targets = np.array(targets)
        else:
            sources = np.nan
            targets = np.nan
        # for each target and source, draw a plot
        if multiprocess:
            process_list = []
            num = 8
            for part_index in range(num):
                targetname = self.target_name[ceil(len(self.target_name)/num)*part_index:ceil(len(self.target_name)/num)*part_index+ceil(len(self.target_name)/num)]
                p = Process(target=self.part_visualization, args=(FLN_list, FLN_titles, file_name, targetname, sources, targets, 'target', draw_axon_arrow))
                p.start()
                process_list.append(p)
            for part_index in range(num):
                sourcename = self.source_name[ceil(len(self.source_name)/num)*part_index:ceil(len(self.source_name)/num)*part_index+ceil(len(self.source_name)/num)]
                p = Process(target=self.part_visualization, args=(FLN_list, FLN_titles, file_name, sourcename, sources, targets, 'source', draw_axon_arrow))
                p.start()
                process_list.append(p)
            for i in process_list:
                p.join()
        else:
            self.part_visualization(FLN_list, FLN_titles, file_name, self.target_name, sources, targets, 'target', draw_axon_arrow)
            self.part_visualization(FLN_list, FLN_titles, file_name, self.source_name, sources, targets, 'source', draw_axon_arrow)


    def do_visualization(self, FLN_data, data_type, draw_axon_arrow=False, multiprocess=True):
        '''Using simultaneously_visualization to get FLN visualization.'''
        
        print('donot use it')
        print(1/0)

        print('----------------FLN visualization----------------')
        mkdir(self.basedir+'/FLN_weight_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow))
        self.simultaneously_visualization([FLN_data, self.FLN_model], ['FLN_data_'+str(len(self.target_name)), 'FLN_model'], file_name=self.basedir+'/FLN_weight_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow)+'/FLN', draw_axon_arrow=draw_axon_arrow, multiprocess=multiprocess)

        FLN_data_topo = pd.DataFrame(np.zeros(np.shape(FLN_data.values)), index=FLN_data.index, columns=FLN_data.columns)
        FLN_data_topo.values[np.nonzero(FLN_data.values)] = 1
        FLN_model_topo = pd.DataFrame(np.zeros(np.shape(self.FLN_model.values)), index=self.FLN_model.index, columns=self.FLN_model.columns)
        FLN_model_topo.values[np.nonzero(self.FLN_model.values)] = 1
        mkdir(self.basedir+'/FLN_topo_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow))
        self.simultaneously_visualization([FLN_data_topo, FLN_model_topo], ['FLN_data_'+str(len(self.target_name)), 'FLN_model'], file_name=self.basedir+'/FLN_topo_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow)+'/FLN', draw_axon_arrow=draw_axon_arrow, multiprocess=multiprocess)


    def draw_correlation(self, target_list, data_FLN_square, model_FLN_square, data_FLN_rectangle, model_FLN_rectangle, filename):
        print('dont use')
        print(1/0)
        '''Draw correlation plot for each target in target_list. Note that nan is possible because if the data contains all 0, the variance is zero.'''
        correlation_list_square = []
        correlation_list_rectangle = []
        for target in target_list:
            correlation_list_square.append(np.corrcoef(data_FLN_square.loc[:, target], model_FLN_square.loc[:, target])[0, 1])
            correlation_list_rectangle.append(np.corrcoef(data_FLN_rectangle.loc[:, target], model_FLN_rectangle.loc[:, target])[0, 1])
        flatten_corr_square = np.corrcoef(data_FLN_square.loc[:, target_list].values.flatten(), model_FLN_square.loc[:, target_list].values.flatten())[0, 1]
        flatten_corr_rectangle = np.corrcoef(data_FLN_rectangle.loc[:, target_list].values.flatten(), model_FLN_rectangle.loc[:, target_list].values.flatten())[0, 1]
        plt.figure(figsize=(12,7))
        plt.plot(range(len(target_list)), correlation_list_square, 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
        plt.plot(range(len(target_list)), correlation_list_rectangle, 'o-', label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
        plt.xticks(range(len(target_list)), target_list, rotation=50, fontsize=8)
        plt.xlabel('target areas')
        plt.ylabel('correlation')
        plt.ylim([-0.2, 1.1])
        plt.legend(loc='upper right')
        plt.savefig(filename+'.png', dpi=300)
        plt.close()

        
        # plt.figure(figsize=(12,7))
        # plt.plot(range(len(target_list)), correlation_list_square, 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
        # plt.xticks(range(len(target_list)), target_list, rotation=50, fontsize=8)
        # plt.xlabel('target areas')
        # plt.ylabel('correlation')
        # plt.ylim([-0.2, 1.1])
        # plt.legend()
        # plt.savefig(filename+'_square.png', dpi=300)
        # plt.close()


        # sort_id = np.argsort(correlation_list_square)
        # plt.figure(figsize=(12,7))
        # plt.plot(range(len(target_list)), np.array(correlation_list_square)[sort_id], 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
        # plt.xticks(range(len(target_list)), np.array(target_list)[sort_id], rotation=50, fontsize=8)
        # plt.xlabel('target areas')
        # plt.ylabel('correlation')
        # plt.ylim([-0.2, 1.1])
        # plt.legend()
        # plt.savefig(filename+'_square_sorted.png', dpi=300)
        # plt.close()


        fig, ax = plt.subplots(figsize=(12,7))
        ax.bar(range(len(target_list)), correlation_list_rectangle, label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
        ax.set_xticks(range(len(target_list)), target_list, rotation=90, fontsize=14)
        ax.set_xlabel('target areas', fontsize=20)
        ax.set_ylabel('correlation', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim([-0.2, 1.1])
        ax.legend(fontsize=16, loc='upper right')
        plt.tight_layout()
        plt.savefig(filename+'_rectangle.png', dpi=300)
        plt.close()


        sort_id = np.argsort(correlation_list_rectangle)
        fig, ax = plt.subplots(figsize=(12,7))
        ax.bar(range(len(target_list)), np.array(correlation_list_rectangle)[sort_id], label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
        ax.set_xticks(range(len(target_list)), np.array(target_list)[sort_id], rotation=90, fontsize=14)
        ax.set_xlabel('target areas', fontsize=20)
        ax.set_ylabel('correlation', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim([-0.2, 1.1])
        ax.legend(fontsize=16, loc='upper right')
        plt.tight_layout()
        plt.savefig(filename+'_rectangle_sorted.png', dpi=300)
        plt.close()
        return correlation_list_square, correlation_list_rectangle, flatten_corr_square, flatten_corr_rectangle


    def do_draw_correlation(self, plot_target_name, FLN_data, FLN_model, plot_name):
        print('dont use')
        print(1/0)
        '''Using draw_correlation to draw correlation plots and get the corr matrix and list.'''
        results = {}
        FLN_data_RECTANGLE = FLN_data[plot_target_name]
        FLN_data_SQUARE = FLN_data.loc[plot_target_name, plot_target_name]
        FLN_model_RECTANGLE = FLN_model[plot_target_name]
        FLN_model_SQUARE = FLN_model.loc[plot_target_name, plot_target_name]
        results['SQUARE_weight'], results['RECTANGLE_weight'], results['flatten_SQUARE_weight'], results['flatten_RECTANGLE_weight']  = self.draw_correlation(plot_target_name, FLN_data_SQUARE, FLN_model_SQUARE, FLN_data_RECTANGLE, FLN_model_RECTANGLE, self.basedir+'/correlation_weight_'+str(plot_name))

        FLN_data_topo = pd.DataFrame(np.zeros(np.shape(FLN_data.values)), index=FLN_data.index, columns=FLN_data.columns)
        FLN_data_topo.values[np.nonzero(FLN_data.values)] = 1
        FLN_data_topo_RECTANGLE = FLN_data_topo[plot_target_name]
        FLN_data_topo_SQUARE = FLN_data_topo.loc[plot_target_name, plot_target_name]
        FLN_model_topo = pd.DataFrame(np.zeros(np.shape(FLN_model.values)), index=FLN_model.index, columns=FLN_model.columns)
        FLN_model_topo.values[np.nonzero(FLN_model.values)] = 1
        FLN_model_topo_RECTANGLE = FLN_model_topo[plot_target_name]
        FLN_model_topo_SQUARE = FLN_model_topo.loc[plot_target_name, plot_target_name]
        results['SQUARE_topo'], results['RECTANGLE_topo'], results['flatten_SQUARE_topo'], results['flatten_RECTANGLE_topo']  = self.draw_correlation(plot_target_name, FLN_data_topo_SQUARE, FLN_model_topo_SQUARE, FLN_data_topo_RECTANGLE, FLN_model_topo_RECTANGLE, self.basedir+'/correlation_topo_'+str(plot_name))
        with open(self.basedir+'/results.txt', 'w') as f:
            f.write(plot_name+'\r\n')
            for key in results.keys():
                f.write(key+str(np.mean(results[key]))+'\r\n')
        return results


    def force_vector_field(self):
        '''Draw a pic showing the force vector field, note that for some cases, there doesn't exist a force vector field.'''
        x_list = np.linspace(self.x_min, self.x_max, self.mesh_dict[self.animal])
        y_list = np.linspace(self.y_min, self.y_max, self.mesh_dict[self.animal])
        sources = []
        for x in x_list:
            for y in y_list:
                if self.whether_in_area([x, y], self.pgon_whole['whole_cortex'])==True and self.polygon_index([[x, y]])[0]!='out_of_areas':
                    sources.append([x, y])
        targets = self.force_vector(sources,self.beta,self.force_centers_dict)
        sources = np.array(sources)
        targets = np.array(targets)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axis('off')
        for i in range(len(sources)):
            source = sources[i]
            target = targets[i]
            if self.animal=='marmoset':
                plt.arrow(source[0], source[1], (target[0]-source[0])*0.5, (target[1]-source[1])*0.5, fc = default_blue, ec = default_blue, head_width = 0.1,head_length = 0.1)
            else:
                plt.arrow(source[0], source[1], target[0]-source[0], target[1]-source[1], fc = default_blue, ec = default_blue, head_width = 0.1,head_length = 0.1)
        for name in self.area_coords.keys():
            boundary = self.area_coords[name]
            boundary = np.append(boundary, [boundary[0]], axis=0)
            plt.plot(boundary.T[0], boundary.T[1], c='r')
        for name in self.force_centers_dict.keys():
            plt.scatter(self.force_centers_dict[name][0], self.force_centers_dict[name][1], s=0.5, c='black')
        mkdir(self.basedir+'/force_vector_field/')
        plt.savefig(self.basedir+'/force_vector_field/force_vector_field_without_name.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
        for name in self.area_coords.keys():
            plt.text(self.name_location[name][0], self.name_location[name][1], name, fontsize=20)
        plt.savefig(self.basedir+'/force_vector_field/force_vector_field.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close()


    def target_location_visualization(self):
        '''Plot the target location (where the injections belong to).'''
        mkdir('../../data/'+str(self.animal)+'/target_location_visualization/')
        file_name = r'../../data/'+str(self.animal)+'/target_location_visualization/target_location_visualization_'+str(len(self.target_name))+'.jpg'
        targets = self.coords_target_name
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axis('off')
        for name in self.all_name_coords:
            boundary = self.area_coords[name]
            boundary = np.append(boundary, [boundary[0]], axis=0)
            plt.plot(boundary.T[0], boundary.T[1], c='black')
            plt.text(self.name_location[name][0], self.name_location[name][1], name, fontsize=20)
        for target in targets:
            plt.fill(self.area_coords[target].T[0], self.area_coords[target].T[1], c='g')
        plt.savefig(str(file_name), bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close()

    
    def do_visualize_connection(self):
        if self.rearranged==False:
            visualize_connection(self.ss, self.indices_ss, self.indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location)
            visualize_connection_with_injection(self.ss, self.tt, self.indices_ss, self.indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location, self.animal)
        else:
            visualize_connection(self.new_ss, self.new_indices_ss, self.new_indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location)


    def degree_hier_plot(self):
        if self.hierarchy_exist:
            mkdir(self.basedir+'/degree_hier_relation/')
            fig, ax = plt.subplots(figsize=(8, 8))
            for area in self.hierarchy.keys():
                ax.scatter(self.hierarchy[area], np.count_nonzero(self.FLN_model.loc[:, area]), color=default_blue)
            ax.set_xlabel('hierarchy')
            ax.set_ylabel('area_level_degree')
            plt.savefig(self.basedir+'/degree_hier_relation/degree_hier_plot.png', dpi=300)
            plt.close()


    def self_connection_plot(self):
        if self.animal=='marmoset':
            clustered_source_name, clustered_target_name = clustered_name(self.animal)
            intrinsic_proportion(self.animal, self.basedir, clustered_target_name, self.connection_count_matrix_with_self_connection)
        if self.animal=='marmoset-V4':
            clustered_source_name, clustered_target_name = clustered_name(self.animal)
            FLNt_data = pd.read_excel(r'../../data/'+str(self.animal)+'/FLNt_data.xlsx', index_col=0)
            FLNt_data.index = FLNt_data.index.astype(str)
            FLNt_data.columns = FLNt_data.columns.astype(str)

            intrinsic_proportion_data = []
            for target in clustered_target_name:
                intrinsic_proportion_data.append(FLNt_data.loc[target, target])

            FLNt_model = self.connection_count_matrix_with_self_connection.copy()
            FLNt_model = FLNt_model / FLNt_model.sum(axis=0)
            intrinsic_proportion_model = []
            for target in clustered_target_name:
                intrinsic_proportion_model.append(FLNt_model.loc[target, target])
            uf.mkdir(self.basedir+'/self_connection_proportion/')
            uf.intrisic_bar_scatter_plot(clustered_target_name, self.basedir, intrinsic_proportion_data, intrinsic_proportion_model, corrected_str='')

    
    def do_rearrange(self):
        print('donot use it')
        print(1/0)
        self.new_ss, self.new_tt, self.new_indices_ss, self.new_indices_tt = rearrange(self.ss, self.tt, self.indices_ss, self.indices_tt, self.animal)
        self.rearranged = True
        np.save(self.basedir+'/full_data/new_ss_'+str(self.na)+'.npy', self.new_ss)
        np.save(self.basedir+'/full_data/new_tt_'+str(self.na)+'.npy', self.new_tt)
        np.save(self.basedir+'/full_data/new_indices_ss_'+str(self.na)+'.npy', self.new_indices_ss)
        np.save(self.basedir+'/full_data/new_indices_tt_'+str(self.na)+'.npy', self.new_indices_tt)

    
    def draw_area_size_degree_plot(self):
        area_size_degree_plot(self.FLN_data, self.FLN_model, self.animal, self.basedir+'/area_size_and_degree/')

    
    def do_FLN_visualization(self):
        if self.animal in ['mouse', 'marmoset', 'macaque', 'marmoset-V4']:
            visualize_fln(self.FLN_model, self.FLN_data, self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='FLN', model_type='FLN', vmax=0.25)
        if self.animal=='marmoset':
            visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)
        if self.animal=='human':
            visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)
            # visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.DTI_data/np.sum(self.DTI_data.values)*20, self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)


    def do_remove_long_range_area_by_area_and_draw_corr(self):
        print('need change, dont use')
        print(1/0)
        distance_matrix_2D = pd.read_excel('../../data/'+str(self.animal)+'/distance_matrix_2D.xlsx', index_col=0)
        remove_long_range_area_by_area_and_draw_corr(self.FLN_model, self.FLN_data, distance_matrix_2D, self.target_name, np.linspace(0, np.max(distance_matrix_2D.values), 30, endpoint=True)[1:], self.basedir)


    def do_remove_long_range_and_cal_corr(self):
        if self.animal!='human':
            remove_long_range_and_cal_corr(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.distance_matrix_2D, self.basedir)
        else:
            remove_long_range_and_cal_corr(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.distance_matrix_2D, self.basedir)
            # remove_long_range_and_cal_corr(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.distance_matrix_2D, self.basedir)


    def do_draw_angle_distance_kde_plot(self):
        if self.animal!='human':
            do_draw_kde_plot(self.animal, self.FLN_data, self.FLN_model, self.basedir)
        else:
            do_draw_kde_plot(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir)
            # do_draw_kde_plot(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir)


    def do_compare_area_by_area(self):
        optimal_para = pd.read_excel('../../results/optimal/optimal_para.xlsx', index_col=0)
        if self.animal=='marmoset' or self.animal=='marmoset-V4':
            threshold = 5 * optimal_para.loc['marmoset', 'Lambda']
        else:
            threshold = 5 * optimal_para.loc[self.animal, 'Lambda']
        if self.animal!='human':
            compare_area_by_area(self.FLN_data, self.FLN_model, self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'FLNe')
            if self.animal!='mouse' and self.animal!='marmoset-V4':
                FLN_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.FLN_data, threshold, self.animal, distance_type='2D')
                compare_area_by_area(FLN_data_without_prefrontal_long_range, self.FLN_model, self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'FLNe')
            if self.animal=='marmoset':
                FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
                FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
                FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
                compare_area_by_area(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')

                FLN_data_without_prefrontal_long_range = remove_prefrontal_long_range(FLN_with_intrinsic_data, threshold, self.animal, distance_type='2D')
                compare_area_by_area(FLN_data_without_prefrontal_long_range, FLN_with_intrinsic_model, self.target_name, 'with_intrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'FLNt')


                FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'connection_density')
                compare_area_by_area(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_corrected_by_connection_density_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')

                FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'neuron_density')
                compare_area_by_area(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_corrected_by_neuron_density_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')


                compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name, 'DTI_extrinsic', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
                # compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.source_name, 'DTI_extrinsic', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
                
                DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
                compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name, 'DTI_extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
                DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
                # compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.source_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
        else:
            compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
            # compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
            
            DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
            compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
            DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
            # compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'dMRI')


    def do_corr_skewness_plot(self):
        if self.animal=='marmoset':
            FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
            FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
            FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
            corr = self.do_area_by_area_correlation()
            corr_with_intrinsic = area_by_area_correlation(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name)
            corr_list = []
            corr_with_intrinsic_list = []
            for target in self.target_name:
                corr_list.append(corr[target])
                corr_with_intrinsic_list.append(corr_with_intrinsic[target])
            corr_skewness_plot(corr_list, corr_with_intrinsic_list, self.animal, self.basedir)


    def do_corr_direction_preference(self):
        if self.animal=='marmoset':
            FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
            FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
            FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
            corr = self.do_area_by_area_correlation()
            corr_with_intrinsic = area_by_area_correlation(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name)
            corr_list = []
            corr_with_intrinsic_list = []
            for target in self.target_name:
                corr_list.append(corr[target])
                corr_with_intrinsic_list.append(corr_with_intrinsic[target])
            corr_direction_preference_plot(self.animal, corr_list, corr_with_intrinsic_list, self.target_name, self.basedir)
        elif self.animal=='human':
            corr = self.do_area_by_area_correlation()
            corr_list = []
            for target in self.target_name:
                corr_list.append(corr[target])
            corr_direction_preference_plot(self.animal, corr_list, np.nan, self.target_name, self.basedir)
        else:
            corr = self.do_area_by_area_correlation()
            corr_list = []
            for target in self.target_name:
                corr_list.append(corr[target])
            corr_direction_preference_plot(self.animal, corr_list, np.nan, self.target_name, self.basedir)


    def rmae(self):
        if self.animal!='human':
            RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.FLN_data, self.FLN_model, self.basedir, 'FLN', self.target_name, self.source_name, self.animal)
        else:
            RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir, 'dMRI', self.target_name, self.source_name, self.animal)
            # RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir, 'dMRI', self.target_name, self.source_name, self.animal)
        if self.animal=='marmoset':
            # 注意这里要用source_name而不是target_name
            RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir, 'dMRI', self.source_name, self.source_name, self.animal)
        return RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link


    def do_error_analysis(self):
        if self.animal!='human':
            error_analysis(self.FLN_data, self.FLN_model, self.basedir+'/error_analysis/', 'FLN', self.target_name, self.source_name, self.animal)
        else:
            error_analysis(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir+'/error_analysis/', 'dMRI', self.target_name, self.source_name, self.animal)
        if self.animal=='marmoset':
            # 注意这里要用source_name而不是target_name
            error_analysis(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir+'/error_analysis_dMRI/', 'dMRI', self.source_name, self.source_name, self.animal)


    def do_neuron_level_distance_model_data(self):
        neuron_level_distance_model_data(self.animal, self.ss, self.tt, self.basedir)


    def do_get_all_corr(self):
        if self.animal!='human':
            get_all_corr(self.FLN_data, self.FLN_model, self.target_name, self.basedir, 'without_intrinsic', self.animal)
        else:
            get_all_corr(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.basedir, 'without_intrinsic', self.animal)
            # get_all_corr(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.basedir, 'without_intrinsic', self.animal)
        if self.animal=='marmoset':
            FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
            FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
            FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
            get_all_corr(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_intrinsic', self.animal)

            FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'connection_density')
            get_all_corr(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_corrected_by_connection_density_intrinsic', self.animal)

            FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'neuron_density')
            get_all_corr(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_corrected_by_neuron_density_intrinsic', self.animal)


            get_all_corr(self.DTI_data, self.symmetry_CCM, self.target_name, self.basedir, 'DTI', self.animal)
        if self.animal=='marmoset-V4':
            FLNt = pd.read_excel('../../data/marmoset-V4/FLNt_data.xlsx', index_col=0)
            FLNt.index = FLNt.index.astype(str)
            FLNt.columns = FLNt.columns.astype(str)
            FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
            FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
            get_all_corr(FLNt, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_intrinsic', self.animal)

    
    def do_occupied_connection(self):
        if self.animal!='human':
            occupied_connection(self.animal, self.FLN_data, self.FLN_model, self.basedir)
        else:
            occupied_connection(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir)
            # occupied_connection(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir)


    def do_visualize_connection_matrix_and_distribution(self):
        if self.animal!='human':
            visualize_connection_matrix_and_distribution(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.source_name, self.basedir, self.distance_matrix_2D, '2D')
            visualize_connection_matrix_and_distribution(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.source_name, self.basedir, self.distance_matrix_3D, '3D')
        else:
            visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.source_name, self.basedir, self.distance_matrix_2D, '2D')
            visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.source_name, self.basedir, self.distance_matrix_3D, '3D')
            # visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.source_name, self.basedir, self.distance_matrix_2D)
        if self.animal=='marmoset':
            # 对于marmoset，单独再画一个DTI的EDR
            edr_model_data(self.source_name, self.target_name, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.animal, self.basedir+'/connection_matrix_and_distribution/EDR_comparison_DTI_2D', '2D')
            edr_model_data(self.source_name, self.target_name, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.animal, self.basedir+'/connection_matrix_and_distribution/EDR_comparison_DTI_3D', '3D')
        if self.animal=='marmoset-V4' or self.animal=='marmoset':
            clustered_source_name, clustered_target_name = clustered_name(self.animal)
            # 定义new_source_name(其实就是对于source_name的重新排序),先放置target_name再放置其余的
            new_source_name = clustered_target_name.copy()
            for area in clustered_source_name:
                if area not in new_source_name:
                    new_source_name.append(area)

            if self.animal=='marmoset':
                FLNt_data = get_FLN_with_intrinsic(self.animal)
            else:
                FLNt_data = pd.read_excel('../../data/marmoset-V4/FLNt_data.xlsx', index_col=0)
            FLNt_data.index = FLNt_data.index.astype(str)
            FLNt_data.columns = FLNt_data.columns.astype(str)

            FLNt_model = self.connection_count_matrix_with_self_connection.copy()
            FLNt_model = FLNt_model.div(FLNt_model.sum(axis=0), axis=1)

            vmax_dict = {'mouse': 0.2, 'macaque': 0.2, 'marmoset': 0.2, 'human': 0.2, 'marmoset-V4': 0.2}
            vmax = vmax_dict[self.animal]

            uf.connection_matrix_heatmap_vertical(FLNt_data, FLNt_model, vmax, 'jet', clustered_target_name, new_source_name, self.basedir+'/connection_matrix_and_distribution/weighted_connection_matrix_t_full', 'normal', self.animal)


    def do_area_by_area_correlation(self):
        '''remove the diagonal of FLN matrix and calculate the correlation'''
        if self.animal=='marmoset':
            # 注意要对source_name做操作
            _ = area_by_area_correlation_without_diagonal(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name)
        if self.animal!='human':
            corr = area_by_area_correlation_without_diagonal(self.FLN_data[self.target_name], self.FLN_model[self.target_name], self.target_name)
        else:
            corr = area_by_area_correlation_without_diagonal(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name)
        return corr
    

    def do_connection_proportion_wrt_to_distance_neuron_level(self):
        connection_proportion_wrt_to_distance_neuron_level(self.ss, self.tt, np.max(self.distance_matrix_2D.values), self.basedir)


    def do_FLN_symmetry(self):
        if self.animal in ['mouse', 'marmoset', 'macaque']:
            FLN_symmetry(self.animal, self.FLN_data, self.FLN_model, self.basedir)


    def do_3D_angle(self):
        if self.animal in ['mouse', 'marmoset', 'macaque']:
            compare_angle_3D_distribution(self.animal, 'center', self.FLN_data, self.FLN_model, self.basedir)
        if self.animal in ['human']:
            compare_angle_3D_distribution(self.animal, 'center', self.DTI_data, self.symmetry_CCM, self.basedir)


    def do_angle_distance_independence(self):
        bin_list = range(10, 30, 1)
        percent_list = range(10, 100, 1)
        for SC_type in ['data', 'model']:
            for bin_type in ['percentile', 'equal']:
                for threshold in [False]:
                    if self.animal!='human':
                        if SC_type=='data':
                            SC = self.FLN_data.copy()
                        if SC_type=='model':
                            SC = self.FLN_model.copy()
                    else:
                        if SC_type=='data':
                            SC = self.DTI_data.copy()
                        if SC_type=='model':
                            SC = self.symmetry_CCM.copy()
                    angle_distance_independence_plots(SC, self.animal, bin_list, bin_type, percent_list, threshold, self.basedir+'/angle_distance_independence/', SC_type)


    def do_cos_similarity(self):
        # cos visualization
        # 得到5 Lambda对应的距离的比例
        Lambda = list(set(self.Lam.values()))[0]
        if len(set(self.Lam.values()))>1:
            print('Lam is not uniform for brain areas')
            print(1/0)
        distance_matrix_2D = self.distance_matrix_2D
        short_range_proportion = distance_matrix_2D[distance_matrix_2D<5*Lambda].count().sum()/(distance_matrix_2D.shape[0]**2)

        if self.animal!='human':
            SC_data = self.FLN_data.copy()
            SC_model = self.FLN_model.copy()
        else:
            SC_data = self.DTI_data.copy()
            SC_data = SC_data/np.sum(SC_data.values)*len(self.source_name)
            SC_model = self.symmetry_CCM.copy()
            SC_model = SC_model/np.sum(SC_model.values)*len(self.source_name)
        cos_visualization(self.animal, self.source_name, SC_model, distance_matrix_2D, 'model', short_range_proportion, self.basedir)
        cos_visualization(self.animal, self.target_name, SC_data, distance_matrix_2D, 'data', short_range_proportion, self.basedir)

# endregion


# region data_process
def weight_matrix_to_topo_matrix(weight_matrix):
    '''
    将权重矩阵转化为拓扑矩阵
    '''
    if isinstance(weight_matrix, pd.DataFrame):
        # 如果输入是DataFrame，将其转换为NumPy数组进行处理
        topo_matrix = (weight_matrix.values > 0).astype(int)
        return pd.DataFrame(topo_matrix, index=weight_matrix.index, columns=weight_matrix.columns)
    
    elif isinstance(weight_matrix, np.ndarray):
        # 如果输入是NumPy数组，直接处理
        return (weight_matrix > 0).astype(int)
    
    else:
        raise ValueError("输入必须是 pandas.DataFrame 或 numpy.ndarray 类型")

def load_SC_basic(filename):
    '''
    读取SC的基本数据
    '''
    SC = pd.read_excel(filename, index_col=0)
    SC.index = SC.index.astype(str)
    SC.columns = SC.columns.astype(str)
    if 'DTI' in filename:
        SC = SC / np.sum(SC.values) * len(SC.index)
    return SC

def load_SC_data(animal, SC_type, rearrange=False, cluster_name=False):
    '''
    rearrange: 是否重新排列SC的顺序,使得被注射的target区域排在前面,其余按顺序排列
    cluster_name: 是否将脑区名字按照cluster重新
    '''
    filename = '../../data/'+str(animal)+'/'+SC_type+'_data.xlsx'
    SC = load_SC_basic(filename)
    SC = rearrange_SC(animal, SC, rearrange, cluster_name)
    return SC

def load_SC_model(animal, SC_type, basedir, rearrange=False, cluster_name=False):
    '''
    rearrange: 是否重新排列SC的顺序,使得被注射的target区域排在前面,其余按顺序排列
    cluster_name: 是否将脑区名字按照cluster重新
    '''
    filename = basedir+'/'+SC_type+'_model.xlsx'
    SC = load_SC_basic(filename)
    SC = rearrange_SC(animal, SC, rearrange, cluster_name)
    return SC

def rearrange_SC(animal, SC, rearrange, cluster_name):
    '''
    重排列SC
    rearrange: 是否重新排列SC的顺序,使得被注射的target区域排在前面,其余按顺序排列
    cluster_name: 是否将脑区名字按照cluster重新
    '''
    if cluster_name:
        clustered_stage_source_name, clustered_stage_target_name = clustered_name(animal)
    else:
        clustered_stage_source_name, clustered_stage_target_name, *args = get_names(animal)
    if rearrange:
        # 定义new_source_name(其实就是对于source_name的重新排序),先放置target_name再放置其余的
        new_source_name = clustered_stage_target_name.copy()
        for area in clustered_stage_source_name:
            if area not in new_source_name:
                new_source_name.append(area)
    else:
        new_source_name = clustered_stage_source_name
    new_SC = cf.reorder_df(SC, new_row=new_source_name, new_col=new_source_name)
    return new_SC

def get_connection_matrix_vmax(animal, SC_type):
    '''
    可视化连接矩阵时的最大值
    '''
    return 0.2

def get_log_c(animal, SC_type):
    '''
    log时的平移系数
    '''
    if animal == 'human':
        return 6
    elif animal == 'marmoset' and SC_type == 'DTI':
        raise NotImplementedError
    else:
        return 7

def get_log_w(animal, SC_type):
    '''
    当SC为0时,设定的基准值
    '''
    return 0.9

def log_SC(SC, animal, SC_type, base=10, apply_c=False, apply_w=False):
    '''
    对SC取log
    '''
    SC_log = np.log(SC) / np.log(base)
    # 处理nan,inf,-inf
    SC_log = SC_log.replace([np.inf, -np.inf], np.nan)
    if apply_c:
        SC_log = SC_log + get_log_c(animal, SC_type)
    if apply_w:
        w = get_log_w(animal, SC_type)
        # 将-inf替换为w
        SC_log = SC_log.replace(-np.inf, w)
        # 将小于w的值设为w
        SC_log[SC_log < w] = w
    return SC_log

def draw_rectangle_for_connection_matrix(ax, target_name, source_name):
    '''
    在连接矩阵上画出矩形,用于区分实验已知部分和位置部分
    '''
    color = cf.GREEN
    linestyle = '--'
    target_num = len(target_name)
    source_num = len(source_name)
    cf.plt_line(ax, x=[0, target_num], y=[0, 0], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[0, 0], y=[0, source_num], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[target_num, target_num], y=[0, source_num], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[0, target_num], y=[source_num, source_num], color=color, linestyle=linestyle)

def draw_mask_for_unmeasured_block(ax, target_name, source_name):
    '''
    在连接矩阵上画出mask,用于区分未测量部分
    '''
    target_num = len(target_name)
    source_num = len(source_name)
    if target_num < source_num:
        color = cf.GRAY
        cmap = cf.get_cmap([color])
        unmeasured_matrix = np.zeros((source_num, source_num))
        for i in range(source_num):
            if i < target_num:
                unmeasured_matrix[:, i] = np.nan
            else:
                unmeasured_matrix[:, i] = 1
        cf.sns_heatmap(ax, unmeasured_matrix, show_xtick=False, show_ytick=False, cmap=cmap, cbar=False)
        cf.add_text(ax, x=(target_num+source_num)/2, y=source_num/2, text='Unknown', fontsize=cf.LABEL_SIZE, ha='center', va='center', transform='data')

def connection_matrix_heatmap_ax(ax_data, ax_model, matrix_data, matrix_model, animal, SC_type, log_matrix=False, topo=False):
    '''
    log_matrix: 是否对矩阵取log
    topo: 是否将矩阵转化为拓扑矩阵
    '''
    target_name, source_name, *args = get_names(animal)

    if log_matrix:
        c = get_log_c(animal, SC_type)
        w = get_log_w(animal, SC_type)
        local_matrix_data = log_SC(SC=matrix_data, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        local_matrix_model = log_SC(SC=matrix_model, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        vmin = w
        vmax = c
        add_geq = False

    else:
        local_matrix_data = matrix_data
        local_matrix_model = matrix_model
        vmin = 0.
        vmax = get_connection_matrix_vmax(animal, SC_type)
        add_geq = True

    if topo:
        cmap = plt.cm.hot
        local_matrix_data = weight_matrix_to_topo_matrix(local_matrix_data)
        local_matrix_model = weight_matrix_to_topo_matrix(local_matrix_model)
        vmin = 0.
        vmax = 1.
    else:
        cmap = plt.cm.jet
    ticks = [vmin, vmax]

    # 连接矩阵
    cbar_kwargs = {'add_geq': add_geq, 'ticks': ticks}
    cf.sns_heatmap(ax_data, local_matrix_data, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    cf.sns_heatmap(ax_model, local_matrix_model, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)

    # 未测量部分
    draw_mask_for_unmeasured_block(ax_data, target_name, source_name)
    
    # 用框指示已测量部分
    draw_rectangle_for_connection_matrix(ax_data, target_name, source_name)
    draw_rectangle_for_connection_matrix(ax_model, target_name, source_name)

    cf.set_ax(ax_data, title=animal+'\nData', xlabel='Target', ylabel='Source')
    cf.set_ax(ax_model, title='Model', xlabel='Target', ylabel='Source')

    matrix_data_to_compare = cf.get_sub_df(matrix_data, col_list=target_name)
    matrix_model_to_compare = cf.get_sub_df(matrix_model, col_list=target_name)
    corr = cf.get_pd_rectangle_corr(matrix_data_to_compare, matrix_model_to_compare, diagonal=False)
    cf.add_text(ax_model, x=0.5, y=1.25, text='Corr='+str(round(corr, 3)), va='center', ha='center', fontsize=cf.LABEL_SIZE)

def multiple_connection_matrix(animal_list, model_dir_list, SC_type_list, figname):
    '''
    同时画多个动物的连接矩阵
    '''
    animal_num = len(animal_list)
    fig, ax = cf.gfa(ncols=animal_num, nrows=2)
    for i, animal, model_dir, SC_type in zip(range(animal_num), animal_list, model_dir_list, SC_type_list):
        SC_data = load_SC_data(animal, SC_type=SC_type, rearrange=True, cluster_name=True)
        SC_model = load_SC_model(animal, SC_type=SC_type, basedir=model_dir, rearrange=True, cluster_name=True)
        connection_matrix_heatmap_ax(ax_data=ax[0][i], ax_model=ax[1][i], matrix_data=SC_data, matrix_model=SC_model, animal=animal, SC_type=SC_type, log_matrix=False, topo=False)
    cf.add_axes_list_tag_by_order(ax[0, :])
    cf.save_fig(fig, figname, formats=['pdf', 'png'], bbox_inches='tight')
# endregion


# region visualization

# endregion