@testset "CUDA utilities                                                      " begin
    @testset "_linear_to_cart" begin
        # 1D
        @test CUDAExt._linear_to_cart(1i32, (5i32,)) == CartesianIndex(1)
        @test CUDAExt._linear_to_cart(3i32, (5i32,)) == CartesianIndex(3)
        @test CUDAExt._linear_to_cart(5i32, (5i32,)) == CartesianIndex(5)

        # 2D
        @test CUDAExt._linear_to_cart(1i32,  (5i32, 4i32)) == CartesianIndex(1, 1)
        @test CUDAExt._linear_to_cart(3i32,  (5i32, 4i32)) == CartesianIndex(3, 1)
        @test CUDAExt._linear_to_cart(5i32,  (5i32, 4i32)) == CartesianIndex(5, 1)
        @test CUDAExt._linear_to_cart(6i32,  (5i32, 4i32)) == CartesianIndex(1, 2)
        @test CUDAExt._linear_to_cart(8i32,  (5i32, 4i32)) == CartesianIndex(3, 2)
        @test CUDAExt._linear_to_cart(10i32, (5i32, 4i32)) == CartesianIndex(5, 2)
        @test CUDAExt._linear_to_cart(11i32, (5i32, 4i32)) == CartesianIndex(1, 3)
        @test CUDAExt._linear_to_cart(13i32, (5i32, 4i32)) == CartesianIndex(3, 3)
        @test CUDAExt._linear_to_cart(15i32, (5i32, 4i32)) == CartesianIndex(5, 3)
        @test CUDAExt._linear_to_cart(16i32, (5i32, 4i32)) == CartesianIndex(1, 4)
        @test CUDAExt._linear_to_cart(18i32, (5i32, 4i32)) == CartesianIndex(3, 4)
        @test CUDAExt._linear_to_cart(20i32, (5i32, 4i32)) == CartesianIndex(5, 4)

        # 3D
        @test CUDAExt._linear_to_cart(1i32,  (5i32, 4i32, 3i32)) == CartesianIndex(1, 1, 1)
        @test CUDAExt._linear_to_cart(5i32,  (5i32, 4i32, 3i32)) == CartesianIndex(5, 1, 1)
        @test CUDAExt._linear_to_cart(6i32,  (5i32, 4i32, 3i32)) == CartesianIndex(1, 2, 1)
        @test CUDAExt._linear_to_cart(10i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 2, 1)
        @test CUDAExt._linear_to_cart(11i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 3, 1)
        @test CUDAExt._linear_to_cart(15i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 3, 1)
        @test CUDAExt._linear_to_cart(16i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 4, 1)
        @test CUDAExt._linear_to_cart(20i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 4, 1)
        @test CUDAExt._linear_to_cart(21i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 1, 2)
        @test CUDAExt._linear_to_cart(25i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 1, 2)
        @test CUDAExt._linear_to_cart(26i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 2, 2)
        @test CUDAExt._linear_to_cart(30i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 2, 2)
        @test CUDAExt._linear_to_cart(31i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 3, 2)
        @test CUDAExt._linear_to_cart(35i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 3, 2)
        @test CUDAExt._linear_to_cart(36i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 4, 2)
        @test CUDAExt._linear_to_cart(40i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 4, 2)
        @test CUDAExt._linear_to_cart(41i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 1, 3)
        @test CUDAExt._linear_to_cart(45i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 1, 3)
        @test CUDAExt._linear_to_cart(46i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 2, 3)
        @test CUDAExt._linear_to_cart(50i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 2, 3)
        @test CUDAExt._linear_to_cart(51i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 3, 3)
        @test CUDAExt._linear_to_cart(55i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 3, 3)
        @test CUDAExt._linear_to_cart(56i32, (5i32, 4i32, 3i32)) == CartesianIndex(1, 4, 3)
        @test CUDAExt._linear_to_cart(60i32, (5i32, 4i32, 3i32)) == CartesianIndex(5, 4, 3)

        # 4D
        @test CUDAExt._linear_to_cart(1i32,   (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 1, 1)
        @test CUDAExt._linear_to_cart(6i32,   (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 1, 1)
        @test CUDAExt._linear_to_cart(11i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 1, 1)
        @test CUDAExt._linear_to_cart(16i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 1, 1)
        @test CUDAExt._linear_to_cart(21i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 2, 1)
        @test CUDAExt._linear_to_cart(26i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 2, 1)
        @test CUDAExt._linear_to_cart(31i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 2, 1)
        @test CUDAExt._linear_to_cart(36i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 2, 1)
        @test CUDAExt._linear_to_cart(41i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 3, 1)
        @test CUDAExt._linear_to_cart(46i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 3, 1)
        @test CUDAExt._linear_to_cart(51i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 3, 1)
        @test CUDAExt._linear_to_cart(56i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 3, 1)
        @test CUDAExt._linear_to_cart(61i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 1, 2)
        @test CUDAExt._linear_to_cart(66i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 1, 2)
        @test CUDAExt._linear_to_cart(71i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 1, 2)
        @test CUDAExt._linear_to_cart(76i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 1, 2)
        @test CUDAExt._linear_to_cart(81i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 2, 2)
        @test CUDAExt._linear_to_cart(86i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 2, 2)
        @test CUDAExt._linear_to_cart(91i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 2, 2)
        @test CUDAExt._linear_to_cart(96i32,  (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 2, 2)
        @test CUDAExt._linear_to_cart(101i32, (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 1, 3, 2)
        @test CUDAExt._linear_to_cart(106i32, (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 2, 3, 2)
        @test CUDAExt._linear_to_cart(111i32, (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 3, 3, 2)
        @test CUDAExt._linear_to_cart(116i32, (5i32, 4i32, 3i32, 2i32)) == CartesianIndex(1, 4, 3, 2)
    end

    @testset "show_tuning_info!" begin
        @test CUDAExt.TUNING_INFO[] == false
        ReSolverFlowsBase.show_tuning_info!(true)
        @test CUDAExt.TUNING_INFO[] == true
        ReSolverFlowsBase.show_tuning_info!(false)
        @test CUDAExt.TUNING_INFO[] == false
    end

    @testset "set_tuning_samples!" begin
        @test CUDAExt.TUNING_SAMPLES[] == 5
        ReSolverFlowsBase.set_tuning_samples!(10)
        @test CUDAExt.TUNING_SAMPLES[] == 10
        ReSolverFlowsBase.set_tuning_samples!(1)
        @test CUDAExt.TUNING_SAMPLES[] == 1
        @test_throws ArgumentError ReSolverFlowsBase.set_tuning_samples!(0)
        @test_throws ArgumentError ReSolverFlowsBase.set_tuning_samples!(-1)
    end

    dummy_kernel1(x, y, z) = nothing
    dummy_kernel2(x, y)    = nothing

    @testset "_get_launch_params" begin
        # initialises as empty
        @test isempty(CUDAExt.LAUNCH_PARAMS)

        # first call adds to dict
        CUDAExt._get_launch_params(dummy_kernel1, CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1))
        @test length(CUDAExt.LAUNCH_PARAMS) == 1

        # second call to same types is invariant
        CUDAExt._get_launch_params(dummy_kernel1, CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1))
        @test length(CUDAExt.LAUNCH_PARAMS) == 1

        # resetting removes single element
        ReSolverFlowsBase.reset_launch_params!()
        @test isempty(CUDAExt.LAUNCH_PARAMS)

        # two different methods
        CUDAExt._get_launch_params(dummy_kernel1, CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1))
        CUDAExt._get_launch_params(dummy_kernel1, CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1, 1))
        CUDAExt._get_launch_params(dummy_kernel2, CUDA.zeros(Float32, 1),
                                                  CUDA.zeros(Float32, 1))
        @test length(CUDAExt.LAUNCH_PARAMS) == 3

        # resetting removes all elements
        ReSolverFlowsBase.reset_launch_params!()
        @test isempty(CUDAExt.LAUNCH_PARAMS)
    end
end
