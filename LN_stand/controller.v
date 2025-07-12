module controller #(
    parameter N = 4,
    parameter WIDTH = 16,
    parameter ADDR_WIDTH = 2
)(
    input wire clk,
    input wire rst,
    input wire en,
    input wire [1:0] inst_vfu_in,
    input wire [2:0] inst_sfu_in,

    output reg [ADDR_WIDTH-1:0] write_addr,
    output reg [ADDR_WIDTH-1:0] read_addr,
    output reg write_enable,
    output reg [1:0] inst_vfu,
    output reg [2:0] inst_sfu,
    output reg data_ready,
    output reg a_vec_sel  // a_vec_mux 선택 신호 추가
);

    localparam IDLE  = 2'b00;
    localparam WRITE = 2'b01;
    localparam READ  = 2'b10;
    localparam DONE  = 2'b11;

    reg [1:0] state, next_state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            write_addr <= 0;
            read_addr <= 0;
            write_enable <= 0;
            inst_vfu <= 0;
            inst_sfu <= 0;
            data_ready <= 0;
            a_vec_sel <= 0;
        end else begin
            state <= next_state;
            case(state)
                WRITE: begin
                    write_enable <= 1;
                    write_addr <= write_addr + 1;
                    a_vec_sel <= 0;  // input_vec 사용 (write 중)
                end
                READ: begin
                    write_enable <= 0;
                    read_addr <= read_addr + 1;
                    a_vec_sel <= 1;  // BRAM 데이터 사용 (read 중)
                end
                default: begin
                    write_enable <= 0;
                    a_vec_sel <= 0;
                end
            endcase
            inst_vfu <= inst_vfu_in;
            inst_sfu <= inst_sfu_in;
        end
    end

    always @(*) begin
        next_state = state;
        data_ready = 0;
        case(state)
            IDLE: if (en) next_state = WRITE;
            WRITE: if (write_addr == N-1) next_state = READ;
            READ: if (read_addr == N-1) begin
                next_state = DONE;
                data_ready = 1;
            end
            DONE: next_state = IDLE;
        endcase
    end

endmodule
