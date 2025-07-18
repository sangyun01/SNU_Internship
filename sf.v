
module softmax(
    input wire clk,
    input wire rst_n,
    input wire [64*16-1:0] x_in,
    input wire x_in_valid, 
    input wire next_ready,
    output wire softmax_ready,
    output wire softmax_valid, 
    output wire [64*16-1:0] softmax
    );
    
    assign softmax_ready=max_ready_all & subexp_a_ready_all;
    
    wire modified_x_in_valid;
    assign modified_x_in_valid=x_in_valid & softmax_ready; 
    
    //1. max
    wire max_valid;
    wire max_ready_all;
    wire [15:0] max;
    FP16_max64 u_max64(
        .clk(clk),
        .rst_n(rst_n),
        .x(x_in),
        .x_valid(modified_x_in_valid),
        .max_ready_all(max_ready_all),
        .next_ready(subexp_b_ready_all), 
        .max_valid(max_valid), 
        .max(max)
    );
    ////////////////////////////////////////////////////////////////
    
    
    //// for 1 : n data transfer ////
    wire [63:0] subexp_a_ready;
    wire [63:0] subexp_b_ready;
    wire subexp_a_ready_all;
    wire subexp_b_ready_all;
    assign subexp_a_ready_all=&subexp_a_ready;
    assign subexp_b_ready_all=&subexp_b_ready;
    wire modified_max_valid;
    assign modified_max_valid=max_valid & subexp_b_ready_all; 
    
    
    //2. sub & exp
    wire [63:0] subexp_valid; //for & operation
    wire [15:0] subexp [0:63];
    wire sum_mult_ready; //input must be located above the 'generate' !!!
    genvar i;
    generate
        for(i=0; i<64; i=i+1)
            FP16_subexp u_subexp(
                .clk(clk),
                .rst_n(rst_n),
                .a(x_in[i*16 +: 16]),
                .b(max),
                .a_valid(modified_x_in_valid),
                .b_valid(modified_max_valid), 
                .a_ready(subexp_a_ready[i]),
                .b_ready(subexp_b_ready[i]),
                .next_ready(sum_mult_ready),
                .subexp_valid(subexp_valid[i]), 
                .subexp(subexp[i])
            );
    endgenerate
    
    wire [64*16-1:0] subexp_flatten;
    generate
        for (i=0; i<64; i=i+1)
            assign subexp_flatten[i*16 +: 16] = subexp[i];
    endgenerate
    
    wire subexp_valids;
    assign subexp_valids=&subexp_valid;
    ////////////////////////////////////////////////////////////////
    
    
    assign sum_mult_ready=sum_ready_all & mult_a_ready_all;
    wire modified_subexp_valid;
    assign modified_subexp_valid=subexp_valids & sum_mult_ready;
    
    
    //3. sum
    wire sum_ready_all;
    wire sum_valid;
    wire [15:0] sum;
    FP16_add64 u_add64(
        .clk(clk),
        .rst_n(rst_n),
        .x(subexp_flatten),
        .x_valid(modified_subexp_valid), 
        .sum_ready_all(sum_ready_all), 
        .next_ready(recip_ready),
        .sum_valid(sum_valid), 
        .sum(sum)
    );
    ////////////////////////////////////////////////////////////////
    
    
    //4. recip
    wire recip_ready;
    wire recip_valid;
    wire [15:0] recip;
    FP16_recip u_recip(
        .aclk(clk),                                
        .s_axis_a_tvalid(sum_valid),
        .s_axis_a_tready(recip_ready),
        .s_axis_a_tdata(sum),
        .m_axis_result_tvalid(recip_valid),
        .m_axis_result_tready(mult_b_ready_all),
        .m_axis_result_tdata(recip)
    );
    ////////////////////////////////////////////////////////////////
    
    
    //// for 1 : n data transfer ////
    wire [63:0] mult_a_ready;
    wire [63:0] mult_b_ready;
    wire mult_a_ready_all;
    wire mult_b_ready_all;
    assign mult_a_ready_all=&mult_a_ready;
    assign mult_b_ready_all=&mult_b_ready;
    wire modified_recip_valid;
    assign modified_recip_valid=recip_valid & mult_b_ready_all;
    
    
    //5. mult
    wire [63:0] mult_valid; //for & operation
    wire [15:0] mult [0:63];
    generate
        for(i=0; i<64; i=i+1)
            FP16_mult u_mult(
                .aclk(clk),
                .s_axis_a_tvalid(modified_subexp_valid),
                .s_axis_a_tready(mult_a_ready[i]),
                .s_axis_a_tdata(subexp[i]),
                .s_axis_b_tvalid(modified_recip_valid),
                .s_axis_b_tready(mult_b_ready[i]),
                .s_axis_b_tdata(recip),
                .m_axis_result_tvalid(mult_valid[i]),
                .m_axis_result_tready(next_ready),
                .m_axis_result_tdata(mult[i])
            );
    endgenerate
    
    wire [64*16-1:0] mult_flatten;
    generate
        for (i=0; i<64; i=i+1)
            assign mult_flatten[i*16 +: 16] = mult[i];
    endgenerate
    
    wire mult_valids;
    assign mult_valids=&mult_valid;
    ////////////////////////////////////////////////////////////////
    
    
    //6. output
    assign softmax_valid=mult_valids;
    assign softmax=mult_flatten;
    
endmodule 


