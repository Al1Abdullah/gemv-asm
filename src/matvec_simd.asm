global matvec_simd

section .text
matvec_simd:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r15, rcx

    xor rbx, rbx                 

.row_loop:
    cmp rbx, r15
    jge .done

    mov rax, rbx
    imul rax, r8
    lea r9, [r12 + rax*4]         

    vxorps ymm0, ymm0, ymm0        
    xor rcx, rcx                     

    mov r10, r8
    and r10, -8                       

.col_loop_vec:
    cmp rcx, r10
    jge .horizontal_sum

    vmovups ymm1, [r9 + rcx*4]       
    vmovups ymm2, [r13 + rcx*4]       
    vfmadd231ps ymm0, ymm1, ymm2       

    add rcx, 8
    jmp .col_loop_vec

.horizontal_sum:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.col_loop_tail:
    cmp rcx, r8
    jge .row_done

    vmovss xmm1, [r9 + rcx*4]
    vmovss xmm2, [r13 + rcx*4]
    vmulss xmm1, xmm1, xmm2
    vaddss xmm0, xmm0, xmm1

    inc rcx
    jmp .col_loop_tail

.row_done:
    vmovss [r14 + rbx*4], xmm0
    inc rbx
    jmp .row_loop

.done:
    vzeroupper                          
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
